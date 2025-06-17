import subprocess

def run_command(cmd):
    subprocess.run(cmd, shell=True, check=True)

def run_setup():
    run_command("poetry lock")
    run_command("poetry install")

if __name__ == "__main__":
    run_setup() 