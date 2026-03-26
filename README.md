To run the docker image connect the telemetry port with browser via http:
docker run -d \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
