# AGENTS.md

## Development Tips

### Development

- Use ./agent_scratchpad as you please to test ideas or write temporary files. This is esppecially useful if you want to run short scripts to get precise computations, or to test temporary outputs or values. Note, for actual production/real-world output files from finished scripts, they should go in ./output_files

### Testing, Execution, and Validation

- Use ./__inenv in order to run python, pip, or any other command that must be in the virtual environment. The setup script has already taken care of ensuring that the environment
is set up, and that __inenv is executable