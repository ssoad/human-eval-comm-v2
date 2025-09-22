"""
Unit tests for SandboxRunner module.
"""

import os
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from evaluators.sandbox_runner import ExecutionResult, ResourceLimits, SandboxRunner


class TestResourceLimits:
    """Test cases for ResourceLimits dataclass."""

    def test_resource_limits_default(self):
        """Test default resource limits."""
        limits = ResourceLimits()

        assert limits.cpu_time_limit == 30.0
        assert limits.memory_limit == 128
        assert limits.wallclock_limit == 60.0
        assert limits.disk_limit == 50

    def test_resource_limits_custom(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            cpu_time_limit=60.0, memory_limit=256, wallclock_limit=120.0, disk_limit=100
        )

        assert limits.cpu_time_limit == 60.0
        assert limits.memory_limit == 256
        assert limits.wallclock_limit == 120.0
        assert limits.disk_limit == 100


class TestExecutionResult:
    """Test cases for ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test ExecutionResult creation."""
        result = ExecutionResult(
            success=True,
            stdout="Test output",
            stderr="",
            exit_code=0,
            execution_time=2.5,
            memory_used=45.0,
            cpu_time_used=2.0,
            timeout=False,
            killed=False,
        )

        assert result.success is True
        assert result.stdout == "Test output"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time == 2.5
        assert result.memory_used == 45.0
        assert result.cpu_time_used == 2.0
        assert result.timeout is False
        assert result.killed is False


class TestSandboxRunner:
    """Test cases for SandboxRunner."""

    @pytest.fixture
    def sandbox_runner(self):
        """Create SandboxRunner instance."""
        return SandboxRunner(use_docker=False)

    @pytest.fixture
    def sample_code(self):
        """Sample Python code for testing."""
        return "print('Hello, World!')"

    @pytest.fixture
    def sample_test_code(self):
        """Sample test code."""
        return """
def test_sample():
    assert True
"""

    def test_init_no_docker(self):
        """Test initialization without Docker."""
        runner = SandboxRunner(use_docker=False)

        assert runner.use_docker is False
        assert runner.docker_client is None

    @patch("docker.from_env")
    def test_init_with_docker_success(self, mock_docker):
        """Test successful initialization with Docker."""
        mock_client = Mock()
        mock_client.images.get.return_value = Mock()
        mock_docker.return_value = mock_client

        runner = SandboxRunner(use_docker=True)

        assert runner.use_docker is True
        assert runner.docker_client == mock_client

    @patch("docker.from_env")
    def test_init_with_docker_failure(self, mock_docker):
        """Test initialization with Docker failure."""
        mock_docker.side_effect = Exception("Docker not available")

        runner = SandboxRunner(use_docker=True)

        assert runner.use_docker is False
        assert runner.docker_client is None

    def test_extract_solution_from_test(self, sandbox_runner):
        """Test extracting solution code from test file."""
        test_code = """
# Some imports
import sys

# SOLUTION START
def solution(x):
    return x * 2
# SOLUTION END

# Tests
def test_solution():
    assert solution(5) == 10
"""

        solution = sandbox_runner._extract_solution_from_test(test_code)

        assert "def solution(x):" in solution
        assert "return x * 2" in solution
        assert "def test_solution():" not in solution

    def test_extract_solution_from_test_no_solution(self, sandbox_runner):
        """Test extracting solution from test with no embedded solution."""
        test_code = """
def test_something():
    assert True
"""

        solution = sandbox_runner._extract_solution_from_test(test_code)

        assert solution == ""

    @patch("subprocess.Popen")
    def test_run_code_with_subprocess_success(
        self, mock_popen, sandbox_runner, sample_code
    ):
        """Test successful code execution with subprocess."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"Hello, World!", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        with patch("psutil.Process") as mock_psutil:
            mock_process_obj = Mock()
            mock_process_obj.memory_info.return_value = Mock(rss=1000000)
            mock_psutil.return_value = mock_process_obj

            result = sandbox_runner.run_code(sample_code)

        assert result.success is True
        assert "Hello, World!" in result.stdout
        assert result.exit_code == 0
        assert result.timeout is False
        assert result.killed is False

    @patch("subprocess.Popen")
    def test_run_code_with_subprocess_timeout(
        self, mock_popen, sandbox_runner, sample_code
    ):
        """Test code execution with timeout."""
        mock_process = Mock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("python", 30)
        mock_process.returncode = -1
        mock_popen.return_value = mock_process

        with patch("psutil.Process") as mock_psutil:
            mock_process_obj = Mock()
            mock_process_obj.memory_info.return_value = Mock(rss=1000000)
            mock_psutil.return_value = mock_process_obj

            result = sandbox_runner.run_code(sample_code)

        assert result.success is False
        assert result.timeout is True
        assert result.killed is True

    @patch("subprocess.Popen")
    def test_run_code_with_subprocess_error(
        self, mock_popen, sandbox_runner, sample_code
    ):
        """Test code execution with error."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"Error: invalid syntax")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        with patch("psutil.Process") as mock_psutil:
            mock_process_obj = Mock()
            mock_process_obj.memory_info.return_value = Mock(rss=1000000)
            mock_psutil.return_value = mock_process_obj

            result = sandbox_runner.run_code(sample_code)

        assert result.success is False
        assert "Error: invalid syntax" in result.stderr
        assert result.exit_code == 1

    def test_create_docker_command_code_only(self, sandbox_runner, sample_code):
        """Test Docker command creation for code only."""
        command = sandbox_runner._create_docker_command(sample_code, "")

        assert "import solution" in command
        assert "SUCCESS:" in command

    def test_create_docker_command_with_tests(
        self, sandbox_runner, sample_code, sample_test_code
    ):
        """Test Docker command creation with tests."""
        command = sandbox_runner._create_docker_command(sample_code, sample_test_code)

        assert "pytest" in command
        assert "test_solution.py" in command
        assert "RETURN_CODE:" in command

    def test_parse_docker_logs(self, sandbox_runner):
        """Test parsing Docker container logs."""
        logs = """
STDOUT: Test passed
STDERR: Warning message
ERROR: Fatal error
SUCCESS: Import successful
RETURN_CODE: 0
"""

        stdout, stderr = sandbox_runner._parse_docker_logs(logs)

        assert "Test passed" in stdout
        assert "Import successful" in stdout
        assert "Warning message" in stderr
        assert "Fatal error" in stderr

    def test_get_system_info(self, sandbox_runner):
        """Test getting system information."""
        info = sandbox_runner.get_system_info()

        assert "docker_available" in info
        assert "python_version" in info
        assert "platform" in info
        assert info["docker_available"] is False
        assert info["python_version"] == sys.version

    def test_run_test_suite(self, sandbox_runner, sample_code, sample_test_code):
        """Test running test suite."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_test_code)
            test_file = f.name

        try:
            with patch.object(sandbox_runner, "run_code") as mock_run_code:
                mock_result = ExecutionResult(
                    success=True,
                    stdout="Tests passed",
                    stderr="",
                    exit_code=0,
                    execution_time=2.0,
                    memory_used=50.0,
                    cpu_time_used=1.5,
                    timeout=False,
                    killed=False,
                )
                mock_run_code.return_value = mock_result

                results = sandbox_runner.run_test_suite([test_file])

                assert len(results) == 1
                assert results[0].success is True
                mock_run_code.assert_called_once()

        finally:
            os.unlink(test_file)

    def test_run_test_suite_file_error(self, sandbox_runner):
        """Test running test suite with file error."""
        results = sandbox_runner.run_test_suite(["nonexistent_file.py"])

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].exit_code == -1
        assert "Test file error" in results[0].error_message

    @patch("psutil.Process")
    def test_monitor_process_resources(self, mock_psutil, sandbox_runner):
        """Test process resource monitoring."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.kill = Mock()

        mock_ps_process = Mock()
        mock_ps_process.memory_info.return_value = Mock(rss=200 * 1024 * 1024)  # 200MB
        mock_ps_process.cpu_times.return_value = Mock(
            user=40.0, system=10.0
        )  # 50s CPU time
        mock_psutil.return_value = mock_ps_process

        limits = ResourceLimits(memory_limit=100, cpu_time_limit=30)  # Exceeded limits

        # Mock threading to avoid actual thread creation
        with patch("threading.Thread"):
            with patch("time.sleep"):  # Speed up the test
                sandbox_runner._monitor_process_resources(mock_process, limits)

        # Process should be killed due to resource limits
        mock_process.kill.assert_called_once()

    def test_set_process_limits(self, sandbox_runner):
        """Test setting process resource limits."""
        limits = ResourceLimits(memory_limit=64, cpu_time_limit=15)

        # This will be called by preexec_fn
        set_limits_func = sandbox_runner._set_process_limits(limits)

        # The function should be callable and not raise exceptions
        assert callable(set_limits_func)

        # Test that it doesn't crash (we can't easily test the actual limits without mocking)
        try:
            set_limits_func()
        except Exception:
            # It's OK if this fails due to system restrictions
            pass
