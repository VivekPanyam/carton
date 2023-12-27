# A dockerfile for builds
# ARG BUILD_TARGET=x86_64
ARG BUILD_TARGET=aarch64

# Start with the manylinux_2_28 image
FROM quay.io/pypa/manylinux_2_28_${BUILD_TARGET}
ARG BUILD_TARGET

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

# Get python 3.10 (+ libs)
RUN yum install -y wget && \
    wget https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-${BUILD_TARGET}-unknown-linux-gnu-install_only.tar.gz -O python.tar.gz && \
    tar -xvf python.tar.gz && \
    yum clean all

ENV PATH="/root/.cargo/bin:/python/bin:${PATH}"
RUN cp /python/lib/libpython3.10.so /usr/local/lib && ldconfig

# Install build requirements
RUN pip3 install toml maturin==0.14.13

# Get sccache
RUN mkdir -p /sccache && \
    cd /sccache && \
    wget https://github.com/mozilla/sccache/releases/download/v0.5.4/sccache-v0.5.4-${BUILD_TARGET}-unknown-linux-musl.tar.gz -O sccache.tar.gz && \
    tar -xvf sccache.tar.gz && \
    mv sccache-v0.5.4-${BUILD_TARGET}-unknown-linux-musl/sccache .

# Install nvm and node
ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION v18.16.1
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
RUN source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION && npm install -g yarn@1.22
ENV NODE_PATH $NVM_DIR/versions/node/$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH