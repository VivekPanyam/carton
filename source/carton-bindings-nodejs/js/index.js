const native = require("./carton_native.node")

const load = async (path, opts = {}) => {
    const inner = await native.load({
        path: path,
        ...opts
    })

    return new Carton(inner)
}

module.exports = {
    load
}

class Carton {
    constructor(inner) {
        this.inner = inner;
    }

    get name() {
        return this.inner.name
    }

    get runner() {
        return this.inner.runner
    }

    async infer(tensors) {
        // TODO: this may be a bit brittle across libraries so refactor if needed

        // Get the buffer, shape, stride, and dtype for each input tensor
        const nativeTensors = {};
        for (const key in tensors) {
            const tensor = tensors[key];

            // Should support `ndarray` and `@stdlib/ndarray`
            let buffer = tensor.data.buffer
            let shape = tensor.shape
            let stride = tensor.stride || tensor.strides
            let dtype = tensor.dtype

            nativeTensors[key] = { buffer, shape, dtype, stride }
        }

        // console.log(nativeTensors)

        // Run the model
        return await native.infer.call(this.inner.handle, nativeTensors)
    }
}