"""
Sandbox Runner Module

Abstracted runner that executes Python snippets/tests under resource limits (CPU, RAM, wallclock),
no network access, and read-only filesystem. Implemented using Docker with fallback to subprocess
with resource limits for local execution.
"""

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import docker
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""

    cpu_time_limit: float = 30.0  # seconds
    memory_limit: int = 128  # MB
    wallclock_limit: float = 60.0  # seconds
    disk_limit: int = 50  # MB


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    memory_used: float
    cpu_time_used: float
    timeout: bool
    killed: bool
    error_message: str = ""


class SandboxRunner:
    """Manages sandboxed execution of Python code with resource limits."""

    def __init__(self, use_docker: bool = True, docker_image: str = "python:3.11-slim"):
        """Initialize sandbox runner."""
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_client = None

        if use_docker:
            try:
                self.docker_client = docker.from_env()
                self._ensure_docker_image()
            except Exception as e:
                logger.warning(f"Docker not available, falling back to subprocess: {e}")
                self.use_docker = False

    def _ensure_docker_image(self):
        """Ensure the required Docker image is available."""
        try:
            self.docker_client.images.get(self.docker_image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image: {self.docker_image}")
            self.docker_client.images.pull(self.docker_image)

    def run_code(
        self,
        code: str,
        test_code: str = "",
        resource_limits: Optional[ResourceLimits] = None,
    ) -> ExecutionResult:
        """Execute code in sandboxed environment."""
        if resource_limits is None:
            resource_limits = ResourceLimits()

        if self.use_docker:
            return self._run_with_docker(code, test_code, resource_limits)
        else:
            return self._run_with_subprocess(code, test_code, resource_limits)

    def _run_with_docker(
        self, code: str, test_code: str, limits: ResourceLimits
    ) -> ExecutionResult:
        """Execute code using Docker container."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write code files
                solution_file = os.path.join(temp_dir, "solution.py")
                test_file = os.path.join(temp_dir, "test_solution.py")

                with open(solution_file, "w") as f:
                    f.write(code)

                if test_code:
                    with open(test_file, "w") as f:
                        f.write(test_code)

                # Create Docker container
                container = self.docker_client.containers.create(
                    self.docker_image,
                    command=[
                        "python",
                        "-c",
                        self._create_docker_command(code, test_code),
                    ],
                    mem_limit=f"{limits.memory_limit}m",
                    cpu_period=100000,
                    cpu_quota=int(limits.cpu_time_limit * 100000),
                    network_disabled=True,
                    read_only=True,
                    tmpfs={"/tmp": "rw,size=100m"},
                    working_dir="/tmp",
                )

                # Start container with timeout
                start_time = time.time()
                container.start()

                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=limits.wallclock_limit)
                    execution_time = time.time() - start_time
                    timeout = False
                    killed = False
                except docker.errors.ContainerTimeout:
                    container.kill()
                    result = {"StatusCode": -1}
                    execution_time = time.time() - start_time
                    timeout = True
                    killed = True

                # Get logs
                logs = container.logs().decode("utf-8")
                stdout, stderr = self._parse_docker_logs(logs)

                # Clean up
                container.remove()

                return ExecutionResult(
                    success=result["StatusCode"] == 0,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result["StatusCode"],
                    execution_time=execution_time,
                    memory_used=0.0,  # Docker doesn't provide detailed memory stats easily
                    cpu_time_used=execution_time,
                    timeout=timeout,
                    killed=killed,
                    error_message=(
                        ""
                        if result["StatusCode"] == 0
                        else "Container execution failed"
                    ),
                )

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                memory_used=0.0,
                cpu_time_used=0.0,
                timeout=False,
                killed=False,
                error_message=f"Docker execution error: {e}",
            )

    def _create_docker_command(self, code: str, test_code: str) -> str:
        """Create the command to run inside Docker container."""
        if test_code:
            return f"""
import sys
import traceback
import time
import psutil
import os

# Set resource limits
def set_limits():
    try:
        import resource
        # Set memory limit
        resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
        # Set CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except:
        pass

set_limits()

# Write solution code
with open('/tmp/solution.py', 'w') as f:
    f.write('''{code}''')

# Write test code
with open('/tmp/test_solution.py', 'w') as f:
    f.write('''{test_code}''')

# Import and run solution
try:
    sys.path.insert(0, '/tmp')
    import solution
    
    # Run tests
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'pytest', '/tmp/test_solution.py', '-v'], 
                          capture_output=True, text=True, timeout=30)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("RETURN_CODE:", result.returncode)
    
except Exception as e:
    print("ERROR:", str(e))
    traceback.print_exc()
    sys.exit(1)
"""
        else:
            return f"""
import sys
import traceback

# Write solution code
with open('/tmp/solution.py', 'w') as f:
    f.write('''{code}''')

# Import and validate solution
try:
    sys.path.insert(0, '/tmp')
    import solution
    print("SUCCESS: Solution imported successfully")
except Exception as e:
    print("ERROR:", str(e))
    traceback.print_exc()
    sys.exit(1)
"""

    def _parse_docker_logs(self, logs: str) -> Tuple[str, str]:
        """Parse Docker container logs into stdout and stderr."""
        lines = logs.split("\n")
        stdout_lines = []
        stderr_lines = []

        for line in lines:
            if line.startswith("STDOUT:"):
                stdout_lines.append(line[7:])
            elif line.startswith("STDERR:"):
                stderr_lines.append(line[7:])
            elif line.startswith("ERROR:"):
                stderr_lines.append(line[6:])
            elif line.startswith("SUCCESS:"):
                stdout_lines.append(line[8:])
            elif line.startswith("RETURN_CODE:"):
                continue  # Skip return code lines
            else:
                stdout_lines.append(line)

        return "\n".join(stdout_lines), "\n".join(stderr_lines)

    def _run_with_subprocess(
        self, code: str, test_code: str, limits: ResourceLimits
    ) -> ExecutionResult:
        """Execute code using subprocess with resource limits."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write code files
                solution_file = os.path.join(temp_dir, "solution.py")
                test_file = os.path.join(temp_dir, "test_solution.py")

                with open(solution_file, "w") as f:
                    f.write(code)

                if test_code:
                    with open(test_file, "w") as f:
                        f.write(test_code)

                # Prepare command
                if test_code:
                    cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
                else:
                    cmd = [
                        sys.executable,
                        "-c",
                        f"import sys; sys.path.insert(0, '{temp_dir}'); import solution",
                    ]

                # Execute with resource monitoring
                start_time = time.time()
                start_memory = psutil.virtual_memory().available

                with self._resource_limited_process(cmd, limits) as process:
                    try:
                        stdout, stderr = process.communicate(
                            timeout=limits.wallclock_limit
                        )
                        execution_time = time.time() - start_time
                        timeout = False
                        killed = False
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                        execution_time = time.time() - start_time
                        timeout = True
                        killed = True

                # Calculate resource usage
                end_memory = psutil.virtual_memory().available
                memory_diff = (start_memory - end_memory) / 1024 / 1024  # MB

                # Report memory usage more clearly:
                # Positive = memory consumed, Negative = memory freed
                memory_used = memory_diff

                return ExecutionResult(
                    success=process.returncode == 0,
                    stdout=stdout.decode("utf-8") if stdout else "",
                    stderr=stderr.decode("utf-8") if stderr else "",
                    exit_code=process.returncode,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    cpu_time_used=execution_time,  # Approximation
                    timeout=timeout,
                    killed=killed,
                    error_message=(
                        ""
                        if process.returncode == 0
                        else f"Process exited with code {process.returncode}"
                    ),
                )

        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                memory_used=0.0,
                cpu_time_used=0.0,
                timeout=False,
                killed=False,
                error_message=f"Subprocess execution error: {e}",
            )

    @contextmanager
    def _resource_limited_process(self, cmd: List[str], limits: ResourceLimits):
        """Context manager for resource-limited process execution."""
        process = None
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._set_process_limits(limits),
            )

            # Start resource monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process_resources,
                args=(process, limits),
                daemon=True,
            )
            monitor_thread.start()

            yield process

        finally:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def _set_process_limits(self, limits: ResourceLimits):
        """Set resource limits for the process."""

        def set_limits():
            try:
                import resource

                # Set memory limit
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (
                        limits.memory_limit * 1024 * 1024,
                        limits.memory_limit * 1024 * 1024,
                    ),
                )
                # Set CPU time limit
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(limits.cpu_time_limit), int(limits.cpu_time_limit)),
                )
            except Exception as e:
                logger.warning(f"Failed to set resource limits: {e}")

        return set_limits

    def _monitor_process_resources(self, process, limits: ResourceLimits):
        """Monitor process resources and kill if limits exceeded."""
        try:
            ps_process = psutil.Process(process.pid)

            while process.poll() is None:
                try:
                    # Check memory usage
                    memory_info = ps_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024

                    if memory_mb > limits.memory_limit:
                        logger.warning(
                            f"Memory limit exceeded: {memory_mb}MB > {limits.memory_limit}MB"
                        )
                        process.kill()
                        break

                    # Check CPU time
                    cpu_times = ps_process.cpu_times()
                    cpu_time = cpu_times.user + cpu_times.system

                    if cpu_time > limits.cpu_time_limit:
                        logger.warning(
                            f"CPU time limit exceeded: {cpu_time}s > {limits.cpu_time_limit}s"
                        )
                        process.kill()
                        break

                    time.sleep(0.1)  # Check every 100ms

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")

    def run_test_suite(
        self, test_files: List[str], resource_limits: Optional[ResourceLimits] = None
    ) -> List[ExecutionResult]:
        """Run multiple test files in sandboxed environment."""
        if resource_limits is None:
            resource_limits = ResourceLimits()

        results = []
        for test_file in test_files:
            try:
                with open(test_file, "r") as f:
                    test_code = f.read()

                # Extract solution code if embedded in test
                solution_code = self._extract_solution_from_test(test_code)

                result = self.run_code(solution_code, test_code, resource_limits)
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to run test file {test_file}: {e}")
                results.append(
                    ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=str(e),
                        exit_code=-1,
                        execution_time=0.0,
                        memory_used=0.0,
                        cpu_time_used=0.0,
                        timeout=False,
                        killed=False,
                        error_message=f"Test file error: {e}",
                    )
                )

        return results

    def _extract_solution_from_test(self, test_code: str) -> str:
        """Extract solution code from test file if embedded."""
        # Look for common patterns where solution code is embedded
        lines = test_code.split("\n")
        solution_lines = []
        in_solution = False

        for line in lines:
            if "# SOLUTION START" in line or "def solution(" in line:
                in_solution = True
            elif "# SOLUTION END" in line:
                break
            elif in_solution:
                solution_lines.append(line)

        if solution_lines:
            return "\n".join(solution_lines)

        # If no embedded solution, return empty string
        return ""

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the sandbox environment."""
        info = {
            "docker_available": self.use_docker and self.docker_client is not None,
            "docker_image": self.docker_image if self.use_docker else None,
            "python_version": sys.version,
            "platform": sys.platform,
        }

        if self.use_docker and self.docker_client:
            try:
                info["docker_version"] = self.docker_client.version()
            except Exception:
                pass

        return info

    def cleanup(self):
        """Clean up resources."""
        if self.docker_client:
            try:
                # Clean up any stopped containers
                containers = self.docker_client.containers.list(
                    all=True, filters={"ancestor": self.docker_image}
                )
                for container in containers:
                    if container.status == "exited":
                        container.remove()
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
