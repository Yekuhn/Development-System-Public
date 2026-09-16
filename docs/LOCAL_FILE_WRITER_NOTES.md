# Local File Writer

This package adds deterministic local file writing for EngineerAgent output.

## Flow

```text
EngineerAgent returns code_output
  ↓
operation.py calls local_file_writer.write_code_output(...)
  ↓
files are written into outputs/<run_id>/workspace/
  ↓
write report is saved under outputs/<run_id>/file_writes/
  ↓
QA receives the write report as verification_artifacts.local_write_report
```

## Engineer schema addition

EngineerAgent now returns:

```json
"code_output": {
  "should_write_to_file": true,
  "files": [
    {
      "path": "src/main.py",
      "content": "...full file content...",
      "write_mode": "create_or_overwrite"
    }
  ],
  "notes": ""
}
```

If no files should be written, the agent must return:

```json
"code_output": {
  "should_write_to_file": false,
  "files": [],
  "notes": "No local files should be written for this task."
}
```

## Safety rules

The writer:

- only writes inside `outputs/<run_id>/workspace/`
- blocks absolute paths
- blocks `..` path traversal
- blocks selected runtime folders such as `.git`, `.venv`, `node_modules`, `outputs`, and `logs`
- does not run shell commands
- returns a structured report for audit and QA

## Updated write lifecycle

```text
EngineerAgent returns code_output
  ↓
operation.py writes the attempt into outputs/<run_id>/.attempts/<task_id>/<attempt>/
  ↓
QA reviews the staged write report
  ↓
if QA marks done, operation.py promotes the same code_output into outputs/<run_id>/workspace/
  ↓
final write report is saved under outputs/<run_id>/file_writes/
```

The real workspace is now a QA-approved output area. Failed Engineer attempts remain auditable under `.attempts/` but do not overwrite the workspace.

## File-scope enforcement

`write_code_output()` now supports:

```python
write_code_output(
    code_output=..., 
    workspace_dir=..., 
    task_id=..., 
    engineer_id=..., 
    allowed_paths=work_item["files_expected"],
    enforce_allowed_paths=True,
)
```

If `files_expected` is present, every generated file path must match that scope. Out-of-scope paths are skipped and the report returns:

```json
{
  "status": "scope_error",
  "scope": {
    "enforced": true,
    "allowed_paths": ["src/app.py"],
    "violations": [{"path": "src/other.py", "reason": "path_outside_task_scope"}]
  }
}
```

`operation.py` treats `scope_error` as a deterministic failure and returns the task to the Engineer with QA/retry context.

### Empty scope behavior

An empty `files_expected` list means the task is not authorized to write files. If an Engineer returns files for that task, the writer returns `scope_error` and skips those files. This forces scope expansion to happen through the Engineering Lead/orchestration layer instead of letting workers silently create unplanned files.
