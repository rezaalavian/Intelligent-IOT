Param(
    [string]$SchemaFile = "infrastructure/kafka/topics/pm25.avsc",
    [string]$Subject = "pm25-value",
    [string]$RegistryUrl = "http://127.0.0.1:8081"
)

Write-Host "Registering schema $SchemaFile to $RegistryUrl as subject $Subject"
if (-Not (Test-Path $SchemaFile)) {
    Write-Error "Schema file not found: $SchemaFile"
    exit 2
}

$schema = Get-Content $SchemaFile -Raw
# Schema Registry expects the schema as a JSON-encoded string value for the `schema` key.
$schema_escaped = $schema -replace '\\', '\\\\' -replace '"', '\\"' -replace "`r`n", '\\n' -replace "`n", '\\n' -replace "`r", '\\n'
$body = '{"schema":"' + $schema_escaped + '"}'

try {
    $url = "$RegistryUrl/subjects/$Subject/versions"
    $headers = @{ Accept = 'application/vnd.schemaregistry.v1+json' }
    $resp = Invoke-RestMethod -Method Post -Uri $url -Body $body -ContentType 'application/vnd.schemaregistry.v1+json' -Headers $headers
    Write-Host "Registered schema version: $($resp.id)"
} catch {
    Write-Error "Schema registration failed: $_"
    exit 3
}
