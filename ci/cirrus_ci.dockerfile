# A dockerfile for cirrusci builds

# Start with the manylinux2014 image
FROM quay.io/pypa/manylinux2014_aarch64

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

# Get python 3.10 (+ libs)
RUN yum install -y wget && \
    wget https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-aarch64-unknown-linux-gnu-install_only.tar.gz -O python.tar.gz && \
    tar -xvf python.tar.gz && \
    yum clean all

ENV PATH="/root/.cargo/bin:/python/bin:${PATH}"
RUN cp /python/lib/libpython3.10.so /usr/local/lib && ldconfig