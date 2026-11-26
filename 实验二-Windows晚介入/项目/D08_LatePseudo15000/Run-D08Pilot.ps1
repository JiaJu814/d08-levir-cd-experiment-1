param([ValidateSet('Status','Prepare','Start')][string]$Mode='Status')
$ErrorActionPreference='Stop'
$python=Join-Path (Split-Path $PSScriptRoot) '.venv\Scripts\python.exe'
$entry=Join-Path $PSScriptRoot 'pilot.py'
$env:PYTHONUTF8='1'
$env:PYTHONUNBUFFERED='1'
if($Mode -eq 'Start'){
    if(-not (Test-Path (Join-Path $PSScriptRoot 'results\preflight.json'))){throw 'Run Prepare first.'}
    if(Test-Path (Join-Path $PSScriptRoot 'results\task.json')){
        & $python -B $entry status
        exit $LASTEXITCODE
    }
    $resultDir=Join-Path $PSScriptRoot 'results'
    $launchId=[Guid]::NewGuid().ToString("N")
    $errorLog=Join-Path $resultDir "launcher-$launchId-error.log"
    $proc=Start-Process -FilePath $python -ArgumentList ('-B "'+$entry+'" run') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $resultDir "launcher-$launchId.log") -RedirectStandardError $errorLog -PassThru
    for($i=0;$i -lt 80;$i++){
        if(Test-Path (Join-Path $resultDir 'task.json')){
            & $python -B $entry status
            exit $LASTEXITCODE
        }
        if($proc.HasExited){throw "Pilot supervisor exited. See $errorLog"}
        Start-Sleep -Milliseconds 150
    }
    throw 'Supervisor has not reported a task identity. Check Status and launcher logs before retrying.'
}
& $python -B $entry $Mode.ToLowerInvariant()
exit $LASTEXITCODE

