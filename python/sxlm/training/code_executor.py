"""Code Executor: Provide verifiable rewards for AZR"""

import subprocess
import tempfile
import os

class CodeExecutor:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def execute(self, code: str, test_cases: list = None) -> dict:
        """
        Execute Python code and return results

        Returns:
            dict with 'success', 'output', 'error'
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            os.unlink(temp_file)

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'output': '', 'error': 'Timeout'}
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}

    def compute_reward(self, result: dict) -> float:
        """Compute reward from execution result"""
        if result['success']:
            return 1.0
        else:
            return -0.5
