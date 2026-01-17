"""
REPL Environment for RLM.
Provides a sandboxed Python execution environment with llm_query function.
"""

import io
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class REPLResult:
    """Result from REPL code execution."""
    stdout: str
    stderr: str
    locals: Dict[str, Any]
    execution_time: float
    success: bool = True
    
    def __str__(self):
        if self.stdout:
            return f"Output:\n{self.stdout}"
        elif self.stderr:
            return f"Error:\n{self.stderr}"
        return "(no output)"
    
    def truncated(self, max_chars: int = 2000) -> str:
        """Return truncated output for model consumption."""
        output = str(self)
        if len(output) > max_chars:
            return output[:max_chars] + f"\n... (truncated, {len(output)} total chars)"
        return output


# Safe builtins - no eval, exec, input, etc.
SAFE_BUILTINS = {
    # Core types and functions
    'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
    'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
    'sorted': sorted, 'reversed': reversed, 'range': range,
    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
    'any': any, 'all': all, 'pow': pow, 'divmod': divmod,
    'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
    'repr': repr, 'ascii': ascii, 'format': format, 'hash': hash, 'id': id,
    'iter': iter, 'next': next, 'slice': slice, 'callable': callable,
    'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr, 'delattr': delattr,
    'dir': dir, 'vars': vars,
    'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
    'complex': complex, 'object': object, 'super': super,
    'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
    '__import__': __import__,
    'open': open,
    
    # Exceptions
    'Exception': Exception, 'BaseException': BaseException,
    'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
    'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
    'StopIteration': StopIteration, 'AssertionError': AssertionError,
    
    # Explicitly blocked
    'eval': None, 'exec': None, 'compile': None, 'input': None,
    'globals': None, 'locals': None,
}


class REPLEnvironment:
    """
    Sandboxed Python REPL environment for RLM.
    
    Provides:
    - `context` variable containing the input data
    - `llm_query(prompt)` function for sub-LLM calls
    - `llm_query_batched(prompts)` for concurrent sub-LLM calls
    - Safe execution of Python code
    """
    
    def __init__(
        self,
        context: Any,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_query_batched_fn: Optional[Callable[[List[str]], List[str]]] = None,
        max_output_chars: int = 5000,
    ):
        """
        Initialize REPL environment.
        
        Args:
            context: The context data (can be string, list, dict, etc.)
            llm_query_fn: Function to call for sub-LLM queries
            llm_query_batched_fn: Function for batched sub-LLM queries
            max_output_chars: Maximum characters to return from output
        """
        self.context = context
        self.max_output_chars = max_output_chars
        self._lock = threading.Lock()
        
        # Create temp directory for file operations
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        
        # Initialize globals with safe builtins
        self.globals = {'__builtins__': SAFE_BUILTINS.copy()}
        self.locals = {}
        
        # Add context to environment
        self._setup_context()
        
        # Add llm_query functions
        if llm_query_fn:
            self.globals['llm_query'] = llm_query_fn
        else:
            self.globals['llm_query'] = lambda x: "[llm_query not configured]"
        
        if llm_query_batched_fn:
            self.globals['llm_query_batched'] = llm_query_batched_fn
        else:
            # Default: sequential execution
            def default_batched(prompts: List[str]) -> List[str]:
                return [self.globals['llm_query'](p) for p in prompts]
            self.globals['llm_query_batched'] = default_batched
        
        # Add FINAL_VAR function
        self.globals['FINAL_VAR'] = self._final_var
        
        # Track sub-LLM calls
        self.sub_llm_calls = 0
    
    def _setup_context(self):
        """Set up context variable in the environment."""
        # Store context as JSON file for large contexts
        context_path = os.path.join(self.temp_dir, "context.json")
        
        if isinstance(self.context, str):
            # String context
            self.locals['context'] = self.context
        elif isinstance(self.context, (dict, list)):
            # Structured context - save as JSON
            with open(context_path, 'w') as f:
                json.dump(self.context, f)
            self.locals['context'] = self.context
        else:
            # Other types - convert to string
            self.locals['context'] = str(self.context)
    
    def _final_var(self, variable_name: str) -> str:
        """Return a variable's value as final answer."""
        variable_name = variable_name.strip().strip('"').strip("'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        return f"Error: Variable '{variable_name}' not found"
    
    @contextmanager
    def _capture_output(self):
        """Thread-safe stdout/stderr capture."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def execute(self, code: str) -> REPLResult:
        """
        Execute Python code in the sandboxed environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            REPLResult with stdout, stderr, and updated locals
        """
        start_time = time.time()
        
        with self._capture_output() as (stdout_buf, stderr_buf):
            try:
                # Split imports from other code
                lines = code.split('\n')
                import_lines = []
                other_lines = []
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                
                # Execute imports in globals
                if import_lines:
                    import_code = '\n'.join(import_lines)
                    exec(import_code, self.globals, self.globals)
                
                # Execute main code
                if other_lines:
                    main_code = '\n'.join(other_lines)
                    combined = {**self.globals, **self.locals}
                    exec(main_code, combined, combined)
                    
                    # Update locals with new variables
                    for key, value in combined.items():
                        if key not in self.globals and not key.startswith('_'):
                            self.locals[key] = value
                
                success = True
                
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
                success = False
        
        execution_time = time.time() - start_time
        
        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            locals=self.locals.copy(),
            execution_time=execution_time,
            success=success
        )
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the environment."""
        return self.locals.get(name)
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def __del__(self):
        self.cleanup()
