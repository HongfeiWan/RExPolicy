"""Unit tests for flywheel checkpoint and run-operation infrastructure."""

from __future__ import annotations

import csv
import json
import os
import random
import socket
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from rexpolicy.flywheel.checkpoint import (
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointManager,
    ManifestMismatchError,
    move_optimizer_state_,
    validate_manifest_compatible,
)
from rexpolicy.flywheel.operations import (
    GpuMonitor,
    GpuPreflightError,
    OperationsError,
    RunOperations,
    collect_runtime_provenance,
    preflight_selected_gpus,
    query_gpu_processes,
    sanitize_for_logging,
)


class TestCheckpointManager(unittest.TestCase):
    @staticmethod
    def _model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = torch.nn.Linear(3, 2, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_atomic_round_trip_restores_all_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manager = CheckpointManager(run_dir)
            model, optimizer = self._model_and_optimizer()
            expected_parameters = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            manifest = {
                "git_commit": "abc123",
                "world_size": 1,
                "task": "reach_green_cap/v1",
                "config": {"max_generations": 10, "learning_rate": 1.0e-3},
            }

            random.seed(17)
            np.random.seed(19)
            torch.manual_seed(23)
            checkpoint = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=5,
                global_optimizer_step=11,
                run_manifest=manifest,
                replay_cursor=7,
                extra_rank_state={"archive_records": 9},
                extra_state={"last_good_generation": 5},
            )
            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(()))

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(100)
            optimizer.state.clear()
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)

            resumed = manager.load(
                dit=model,
                optimizer=optimizer,
                current_manifest={
                    **manifest,
                    "config": {"max_generations": 20, "learning_rate": 1.0e-3},
                },
            )
            self.assertEqual(checkpoint.name, "generation-000005")
            self.assertTrue((checkpoint / "COMPLETE").is_file())
            self.assertFalse((manager.checkpoints_dir / ".partial-generation-000005").exists())
            self.assertEqual(
                json.loads((manager.checkpoints_dir / "latest.json").read_text())[
                    "checkpoint"
                ],
                checkpoint.name,
            )
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_parameters[name])
            self.assertEqual(resumed.next_generation, 6)
            self.assertEqual(resumed.global_optimizer_step, 11)
            self.assertEqual(resumed.replay_cursor, 7)
            self.assertEqual(resumed.rank_state["archive_records"], 9)
            self.assertEqual(resumed.extra_state["last_good_generation"], 5)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)
            self.assertTrue(optimizer.state)
            self.assertTrue(
                all(
                    not value.is_cuda
                    for state in optimizer.state.values()
                    for value in state.values()
                    if torch.is_tensor(value)
                )
            )

    def test_partial_directories_are_never_resume_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            partial = manager.checkpoints_dir / ".partial-generation-999999"
            partial.mkdir(parents=True)
            (partial / "COMPLETE").write_text("{}")
            self.assertIsNone(manager.latest_checkpoint())

    def test_two_rank_commit_contains_every_rank_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model, optimizer = self._model_and_optimizer()
            barrier = threading.Barrier(2)
            errors: list[BaseException] = []
            results: list[Path] = []

            def save_rank(rank: int) -> None:
                try:
                    manager = CheckpointManager(
                        Path(temporary),
                        rank=rank,
                        world_size=2,
                        barrier=barrier.wait,
                    )
                    results.append(
                        manager.save(
                            dit=model if rank == 0 else None,
                            optimizer=optimizer if rank == 0 else None,
                            completed_generation=2,
                            global_optimizer_step=3,
                            run_manifest={"git_commit": "a", "world_size": 2},
                            replay_cursor=rank + 10,
                            rng_state={
                                "python": random.getstate(),
                                "numpy": np.random.get_state(),
                                "torch_cpu": torch.get_rng_state(),
                                "torch_cuda": None,
                            },
                        )
                    )
                except BaseException as error:
                    errors.append(error)

            threads = [
                threading.Thread(target=save_rank, args=(rank,)) for rank in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            checkpoint = results[0]
            self.assertTrue((checkpoint / "ranks" / "rank-00000.pt").is_file())
            self.assertTrue((checkpoint / "ranks" / "rank-00001.pt").is_file())
            state = CheckpointManager(Path(temporary)).verify(checkpoint)
            self.assertEqual(state["world_size"], 2)

    def test_manifest_validation_is_strict_except_declared_operations(self) -> None:
        saved = {
            "git_commit": "one",
            "config": {
                "max_generations": 10,
                "learning_rate": 3.0e-6,
                "checkpoint_every": 5,
                "heartbeat_interval_seconds": 30.0,
            },
        }
        validate_manifest_compatible(
            saved,
            {
                "git_commit": "one",
                "config": {
                    "max_generations": 20,
                    "learning_rate": 3.0e-6,
                    "checkpoint_every": 5,
                    "heartbeat_interval_seconds": 10.0,
                },
            },
        )
        with self.assertRaisesRegex(ManifestMismatchError, "cannot decrease"):
            validate_manifest_compatible(
                saved,
                {
                    "git_commit": "one",
                    "config": {
                        "max_generations": 9,
                        "learning_rate": 3.0e-6,
                        "checkpoint_every": 5,
                        "heartbeat_interval_seconds": 30.0,
                    },
                },
            )
        with self.assertRaisesRegex(ManifestMismatchError, "checkpoint_every"):
            validate_manifest_compatible(
                saved,
                {
                    "git_commit": "one",
                    "config": {
                        "max_generations": 20,
                        "learning_rate": 3.0e-6,
                        "checkpoint_every": 10,
                        "heartbeat_interval_seconds": 30.0,
                    },
                },
            )
        with self.assertRaisesRegex(ManifestMismatchError, "learning_rate"):
            validate_manifest_compatible(
                saved,
                {
                    "git_commit": "one",
                    "config": {
                        "max_generations": 20,
                        "learning_rate": 1.0e-5,
                        "checkpoint_every": 5,
                        "heartbeat_interval_seconds": 30.0,
                    },
                },
            )

    def test_rejected_candidate_does_not_replace_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            model, optimizer = self._model_and_optimizer()
            accepted = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=0,
                global_optimizer_step=0,
                run_manifest={"git_commit": "a"},
            )
            rejected = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=5,
                global_optimizer_step=1,
                run_manifest={"git_commit": "a"},
                update_latest=False,
                accepted=False,
            )
            self.assertTrue((rejected / "COMPLETE").is_file())
            self.assertEqual(manager.latest_checkpoint(), accepted)
            (manager.checkpoints_dir / "latest.json").write_text(
                "{corrupt",
                encoding="utf-8",
            )
            self.assertEqual(manager.latest_checkpoint(), accepted)
            with self.assertRaisesRegex(
                CheckpointIntegrityError,
                "rejected candidate",
            ):
                manager.load(
                    dit=model,
                    optimizer=optimizer,
                    current_manifest={"git_commit": "a"},
                    checkpoint=rejected,
                )

    def test_stale_valid_pointer_cannot_hide_newer_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            model, optimizer = self._model_and_optimizer()
            older = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=0,
                global_optimizer_step=0,
                run_manifest={"git_commit": "a"},
            )
            newer = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=5,
                global_optimizer_step=1,
                run_manifest={"git_commit": "a"},
            )
            (manager.checkpoints_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "checkpoint": older.name,
                        "completed_generation": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(manager.latest_checkpoint(), newer)

    def test_checksum_corruption_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            model, optimizer = self._model_and_optimizer()
            checkpoint = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=1,
                global_optimizer_step=1,
                run_manifest={"git_commit": "a"},
            )
            with (checkpoint / "optimizer.pt").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(CheckpointIntegrityError, "Checksum mismatch"):
                manager.verify(checkpoint)

    def test_corrupt_newest_complete_checkpoint_cannot_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            model, optimizer = self._model_and_optimizer()
            manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=0,
                global_optimizer_step=0,
                run_manifest={"git_commit": "a"},
            )
            newest = manager.save(
                dit=model,
                optimizer=optimizer,
                completed_generation=5,
                global_optimizer_step=1,
                run_manifest={"git_commit": "a"},
            )
            with (newest / "checkpoint_state.json").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaises(CheckpointIntegrityError):
                manager.latest_checkpoint()

    def test_non_fp32_dit_is_rejected_without_committing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(Path(temporary))
            model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
            optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaises(CheckpointError):
                manager.save(
                    dit=model,
                    optimizer=optimizer,
                    completed_generation=1,
                    global_optimizer_step=0,
                    run_manifest={"git_commit": "a"},
                )
            self.assertIsNone(manager.latest_checkpoint())

    def test_optimizer_state_move_helper_handles_nested_values(self) -> None:
        model, optimizer = self._model_and_optimizer()
        move_optimizer_state_(optimizer, "cpu")
        self.assertTrue(
            all(
                value.device.type == "cpu"
                for state in optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            )
        )


class TestRunOperations(unittest.TestCase):
    @staticmethod
    def _nvidia_smi(command: list[str] | tuple[str, ...]) -> str:
        joined = " ".join(command)
        if "--query-gpu=index,uuid,name,memory.free,memory.total" in joined:
            return "0, GPU-a, RTX 5090, 30000, 32607\n1, GPU-b, RTX 5090, 31000, 32607\n"
        if "--query-compute-apps" in joined:
            return ""
        if "--query-gpu=index,uuid,name,utilization.gpu" in joined:
            return (
                "0, GPU-a, RTX 5090, 95, 24000, 8000, 32607, 500, 70, 2400\n"
                "1, GPU-b, RTX 5090, 90, 23000, 9000, 32607, 480, 68, 2350\n"
            )
        raise AssertionError(command)

    def test_manifest_metrics_pid_stop_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            operations = RunOperations(run_dir, rank=0, world_size=2)
            operations.write_manifest(
                config={"learning_rate": 3.0e-6, "api_token": "do-not-log"},
                command=[
                    "python",
                    "runner.py",
                    "--password",
                    "synthetic-fixture-password",
                ],
                environment={
                    "CUDA_VISIBLE_DEVICES": "0,1",
                    "NCCL_DEBUG": "INFO",
                    "UNRELATED_PASSWORD": "also-secret",
                },
            )
            operations.metric(
                "generation_complete", generation=1, authorization="bearer secret"
            )
            operations.log(
                "checkpoint saved password=secret", access_token="another-secret"
            )
            operations.update_heartbeat(status="collecting", generation=2)
            operations.request_stop("operator request token=secret")
            operations.mark_exit(
                status="stopped", reason="STOP requested", completed_generation=2
            )
            self.assertTrue((run_dir / "RUNNING.lock").is_file())
            operations.close()

            durable = "\n".join(
                path.read_text(encoding="utf-8")
                for path in run_dir.rglob("*")
                if path.is_file()
            )
            self.assertNotIn("do-not-log", durable)
            self.assertNotIn("synthetic-fixture-password", durable)
            self.assertNotIn("also-secret", durable)
            self.assertNotIn("another-secret", durable)
            self.assertIn("<redacted>", durable)
            self.assertTrue((run_dir / "pids" / "rank-00000.json").is_file())
            self.assertTrue((run_dir / "heartbeats" / "rank-00000.json").is_file())
            launch_records = list((run_dir / "launches").glob("launch-*.json"))
            self.assertEqual(len(launch_records), 1)
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(
                manifest["launch_record"],
                str(launch_records[0].relative_to(run_dir)),
            )
            self.assertTrue(operations.stop_requested())
            self.assertFalse((run_dir / "RUNNING.lock").exists())
            metric = json.loads((run_dir / "metrics.jsonl").read_text())
            self.assertEqual(metric["event"], "generation_complete")

    def test_non_main_rank_does_not_write_shared_metrics_or_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            operations = RunOperations(run_dir, rank=1, world_size=2)
            self.assertIsNone(operations.write_manifest(config={}))
            operations.metric("ignored", value=1)
            operations.close()
            self.assertFalse((run_dir / "run_manifest.json").exists())
            self.assertFalse((run_dir / "metrics.jsonl").exists())

    def test_run_lock_refuses_a_second_rank_zero_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = RunOperations(Path(temporary), rank=0, world_size=1)
            try:
                with self.assertRaisesRegex(OperationsError, "already locked"):
                    RunOperations(Path(temporary), rank=0, world_size=1)
            finally:
                first.close()

    def test_stale_same_host_run_lock_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "RUNNING.lock").write_text(
                json.dumps(
                    {
                        "pid": 2**30,
                        "hostname": socket.gethostname(),
                    }
                )
            )
            operations = RunOperations(run_dir, rank=0, world_size=1)
            try:
                lock = json.loads((run_dir / "RUNNING.lock").read_text())
                self.assertEqual(lock["pid"], os.getpid())
            finally:
                operations.close()

    def test_gpu_monitor_writes_machine_readable_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            monitor = GpuMonitor(
                Path(temporary),
                command_runner=self._nvidia_smi,
                user_lookup=lambda _pid: "test",
            )
            monitor.sample_once()
            with (Path(temporary) / "gpu_metrics.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["uuid"], "GPU-a")
            self.assertEqual(float(rows[0]["utilization_gpu_percent"]), 95.0)

    def test_gpu_preflight_accepts_idle_selected_devices(self) -> None:
        result = preflight_selected_gpus(
            [0, "GPU-b"],
            command_runner=self._nvidia_smi,
            user_lookup=lambda _pid: "test",
        )
        self.assertEqual([device.index for device in result.devices], [0, 1])

    def test_gpu_preflight_rejects_foreign_process_without_killing_it(self) -> None:
        def busy_nvidia_smi(command: list[str] | tuple[str, ...]) -> str:
            if "--query-compute-apps" in " ".join(command):
                return "GPU-a, 424242, python, 6477\n"
            return self._nvidia_smi(command)

        with self.assertRaisesRegex(GpuPreflightError, "pid=424242"):
            preflight_selected_gpus(
                [0],
                command_runner=busy_nvidia_smi,
                user_lookup=lambda _pid: "other-user",
            )

    def test_gpu_preflight_rejects_unlisted_same_process_group_pid(self) -> None:
        child = subprocess.Popen(["sleep", "30"])
        try:
            self.assertEqual(os.getpgid(child.pid), os.getpgrp())

            def busy_nvidia_smi(
                command: list[str] | tuple[str, ...],
            ) -> str:
                if "--query-compute-apps" in " ".join(command):
                    return f"GPU-a, {child.pid}, python, 512\n"
                return self._nvidia_smi(command)

            with self.assertRaisesRegex(GpuPreflightError, f"pid={child.pid}"):
                preflight_selected_gpus(
                    [0],
                    command_runner=busy_nvidia_smi,
                    user_lookup=lambda _pid: "same-group",
                )
        finally:
            child.terminate()
            child.wait(timeout=5)

    def test_gpu_process_query_fails_closed_when_nvidia_smi_fails(self) -> None:
        def failed_nvidia_smi(_command: list[str] | tuple[str, ...]) -> str:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd="nvidia-smi",
                output="",
            )

        with self.assertRaisesRegex(
            OperationsError,
            "Failed to query active NVIDIA compute processes",
        ):
            query_gpu_processes(failed_nvidia_smi)

    def test_recursive_sanitizer_handles_keys_urls_and_cli_pairs(self) -> None:
        sanitized = sanitize_for_logging(
            {
                "token": "one",
                "command": [
                    "--api-key",
                    "two",
                    "https://user:three@example.test/path",
                ],
            }
        )
        self.assertEqual(sanitized["token"], "<redacted>")
        self.assertEqual(sanitized["command"][1], "<redacted>")
        self.assertNotIn("three", sanitized["command"][2])
        self.assertEqual(
            sanitize_for_logging(["sshpass", "-p", "four", "ssh"])[2],
            "<redacted>",
        )
        self.assertEqual(sanitize_for_logging({"tokenizer": "kept"})["tokenizer"], "kept")

    def test_runtime_provenance_collects_git_and_gpu_topology(self) -> None:
        def command_runner(command: list[str] | tuple[str, ...]) -> str:
            joined = " ".join(command)
            if "rev-parse HEAD" in joined:
                return "abc123\n"
            if "branch --show-current" in joined:
                return "feature\n"
            if "status --short" in joined:
                return ""
            if "topo -m" in joined:
                return "GPU0 GPU1 CPU Affinity\n"
            if "--query-gpu=index,uuid,name,driver_version" in joined:
                return "0, GPU-a, RTX 5090, 600.0\n"
            raise AssertionError(command)

        provenance = collect_runtime_provenance(
            repo_root=Path("/repo"), command_runner=command_runner
        )
        self.assertEqual(provenance["git"]["commit"], "abc123")
        self.assertIn("GPU0", provenance["gpu_topology"])


if __name__ == "__main__":
    unittest.main()
