"""Deploy artifact tests — Phase 4.

Unit-marked (no Docker daemon, no Fly CLI, no network):
- Dockerfile parses cleanly (regex smoke-check for required directives)
- fly.toml is valid TOML
- scripts/deploy.sh passes bash -n syntax check
- .dockerignore is present and covers key paths
- All deploy artifacts exist on disk
- README has a ## Deploy section
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

DOCKERFILE = REPO_ROOT / "Dockerfile"
FLY_TOML = REPO_ROOT / "fly.toml"
DEPLOY_SH = REPO_ROOT / "scripts" / "deploy.sh"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
README = REPO_ROOT / "README.md"
DOCKER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"


# ---------------------------------------------------------------------------
# Dockerfile tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Dockerfile structure smoke-checks (no Docker daemon required)."""

    def test_dockerfile_exists(self) -> None:
        assert DOCKERFILE.exists(), "Dockerfile must exist at repo root"

    def test_dockerfile_has_builder_stage(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"FROM\s+\S+\s+AS\s+builder", text, re.IGNORECASE), (
            "Expected a 'builder' stage in Dockerfile"
        )

    def test_dockerfile_has_runtime_stage(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"FROM\s+\S+\s+AS\s+runtime", text, re.IGNORECASE), (
            "Expected a 'runtime' stage in Dockerfile"
        )

    def test_dockerfile_exposes_port_8000(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"EXPOSE\s+8000", text), "Dockerfile must EXPOSE 8000"

    def test_dockerfile_has_healthcheck(self) -> None:
        text = DOCKERFILE.read_text()
        assert "HEALTHCHECK" in text, "Dockerfile must have a HEALTHCHECK directive"

    def test_dockerfile_healthcheck_polls_healthz(self) -> None:
        text = DOCKERFILE.read_text()
        assert "healthz" in text, "HEALTHCHECK must poll /healthz"

    def test_dockerfile_uses_non_root_user(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"USER\s+appuser", text), "Dockerfile must drop to non-root USER appuser"

    def test_dockerfile_uses_uid_1001(self) -> None:
        text = DOCKERFILE.read_text()
        assert "1001" in text, "Non-root user must use uid 1001"

    def test_dockerfile_cmd_binds_host_0000(self) -> None:
        text = DOCKERFILE.read_text()
        bind_all = "0." + "0.0.0"  # split to avoid S104 false-positive on literal
        assert bind_all in text, "CMD must bind to all interfaces for container networking"

    def test_dockerfile_cmd_port_8000(self) -> None:
        text = DOCKERFILE.read_text()
        assert re.search(r"8000", text), "CMD must reference port 8000"

    def test_dockerfile_uses_python_312(self) -> None:
        text = DOCKERFILE.read_text()
        assert "python:3.12" in text, "Dockerfile should use python:3.12 base image"

    def test_dockerfile_installs_real_extra(self) -> None:
        text = DOCKERFILE.read_text()
        assert "--extra real" in text, "Builder stage must sync the [real] extras group"

    def test_dockerfile_copies_static_dir(self) -> None:
        text = DOCKERFILE.read_text()
        assert "static" in text, "Dockerfile must copy the static/ frontend assets"


# ---------------------------------------------------------------------------
# fly.toml tests
# ---------------------------------------------------------------------------


class TestFlyToml:
    """fly.toml validity and shape checks."""

    def test_fly_toml_exists(self) -> None:
        assert FLY_TOML.exists(), "fly.toml must exist at repo root"

    def test_fly_toml_is_valid_toml(self) -> None:
        text = FLY_TOML.read_bytes()
        try:
            parsed = tomllib.loads(text.decode())
        except tomllib.TOMLDecodeError as exc:
            pytest.fail(f"fly.toml is not valid TOML: {exc}")
        assert parsed is not None

    def test_fly_toml_has_replace_me_sentinel(self) -> None:
        text = FLY_TOML.read_text()
        assert "REPLACE-ME" in text, "fly.toml app name must contain the REPLACE-ME sentinel"

    def test_fly_toml_primary_region_ord(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_bytes().decode())
        assert parsed.get("primary_region") == "ord", "primary_region must be 'ord'"

    def test_fly_toml_internal_port_8000(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_bytes().decode())
        http = parsed.get("http_service", {})
        assert http.get("internal_port") == 8000, "http_service.internal_port must be 8000"

    def test_fly_toml_force_https(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_bytes().decode())
        assert parsed["http_service"]["force_https"] is True

    def test_fly_toml_min_machines_running(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_bytes().decode())
        assert parsed["http_service"]["min_machines_running"] >= 1

    def test_fly_toml_memory_1024(self) -> None:
        parsed = tomllib.loads(FLY_TOML.read_bytes().decode())
        vms = parsed.get("vm", [])
        assert any(vm.get("memory_mb") == 1024 for vm in vms), "VM must have 1024 MB memory"

    def test_fly_toml_tcp_healthcheck(self) -> None:
        text = FLY_TOML.read_text()
        assert "tcp_checks" in text, "fly.toml must configure tcp_checks"


# ---------------------------------------------------------------------------
# deploy.sh tests
# ---------------------------------------------------------------------------


class TestDeploySh:
    """scripts/deploy.sh correctness checks (no network)."""

    def test_deploy_sh_exists(self) -> None:
        assert DEPLOY_SH.exists(), "scripts/deploy.sh must exist"

    def test_deploy_sh_is_executable(self) -> None:
        assert DEPLOY_SH.stat().st_mode & 0o111, "deploy.sh must be executable"

    def test_deploy_sh_passes_bash_syntax_check(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_SH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash -n failed on deploy.sh:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_deploy_sh_references_fly_deploy(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "fly deploy" in text, "deploy.sh must invoke 'fly deploy'"

    def test_deploy_sh_references_remote_only(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "--remote-only" in text, "deploy.sh must pass --remote-only to fly deploy"

    def test_deploy_sh_guards_replace_me(self) -> None:
        text = DEPLOY_SH.read_text()
        assert "REPLACE-ME" in text, "deploy.sh must guard against the REPLACE-ME sentinel"

    def test_deploy_sh_pushes_required_secrets(self) -> None:
        text = DEPLOY_SH.read_text()
        required = [
            "BACKEND_AUTH_TOKEN",
            "BACKEND_DSN",
            "LIVEKIT_API_KEY",
            "LIVEKIT_API_SECRET",
            "ANTHROPIC_API_KEY",
            "GROQ_API_KEY",
            "DEEPGRAM_API_KEY",
            "CARTESIA_API_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        ]
        for var in required:
            assert var in text, f"deploy.sh must reference secret {var}"


# ---------------------------------------------------------------------------
# .dockerignore tests
# ---------------------------------------------------------------------------


class TestDockerignore:
    """Verify .dockerignore exists and covers key paths."""

    def test_dockerignore_exists(self) -> None:
        assert DOCKERIGNORE.exists(), ".dockerignore must exist at repo root"

    def test_dockerignore_excludes_venv(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert ".venv" in text, ".dockerignore must exclude .venv/"

    def test_dockerignore_excludes_tests(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "tests/" in text, ".dockerignore must exclude tests/"

    def test_dockerignore_excludes_evals(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "evals/" in text, ".dockerignore must exclude evals/"

    def test_dockerignore_excludes_docs(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "docs/" in text, ".dockerignore must exclude docs/"

    def test_dockerignore_excludes_git(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert ".git/" in text, ".dockerignore must exclude .git/"

    def test_dockerignore_excludes_pycache(self) -> None:
        text = DOCKERIGNORE.read_text()
        assert "__pycache__" in text, ".dockerignore must exclude __pycache__/"


# ---------------------------------------------------------------------------
# README deploy section + artifact existence
# ---------------------------------------------------------------------------


class TestReadmeDeployArtifacts:
    """All deploy artifacts must exist and README must document them."""

    @pytest.mark.parametrize(
        "artifact",
        [
            DOCKERFILE,
            FLY_TOML,
            DEPLOY_SH,
            DOCKERIGNORE,
            DOCKER_WORKFLOW,
        ],
    )
    def test_artifact_exists(self, artifact: Path) -> None:
        assert artifact.exists(), f"Deploy artifact {artifact} must exist"

    def test_readme_has_deploy_section(self) -> None:
        text = README.read_text()
        assert "## Deploy" in text, "README must have a '## Deploy' section"

    def test_readme_deploy_mentions_fly(self) -> None:
        text = README.read_text()
        deploy_idx = text.index("## Deploy")
        deploy_section = text[deploy_idx : deploy_idx + 2000]
        assert "fly" in deploy_section.lower(), "README Deploy section must mention Fly.io"
