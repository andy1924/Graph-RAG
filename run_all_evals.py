import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n[+] Running {script_path}...")
    try:
        # Inject PYTHONIOENCODING env to prevent cp1252 errors on Windows print()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        
        if process.returncode != 0:
             print(f"❌ Script failed with exit code {process.returncode}")
             return False
        return True
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return False

# 1. Run GraphRAG evaluation
run_script("experiments/comprehensive_evaluation.py")

# 2. Run NaiveRAG evaluation
run_script("experiments/naiverag_evaluation.py")

# 3. Run Significance Analysis
run_script("experiments/significance_analysis.py")

print("\n[+] All evaluations complete.")
