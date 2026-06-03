Param()
Write-Host "Starting infra via docker-compose..."
docker-compose up -d --build
Write-Host "Infra started (detached). Use docker-compose logs -f to follow services."
