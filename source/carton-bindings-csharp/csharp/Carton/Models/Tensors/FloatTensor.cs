namespace Carton.Models.Tensors;

using Native;

public class FloatTensor : GenericTensor<float>
{
    public FloatTensor(string key, params ulong[] shape) : base(key, shape)
    {
    }
}