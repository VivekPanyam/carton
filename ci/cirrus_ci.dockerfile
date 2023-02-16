# A dockerfile for cirrusci builds

# Start with the official rust image
FROM rust:latest

# Install python3 dev
RUN apt-get update && apt-get install -y python3-dev && rm -rf /var/lib/apt/lists/*