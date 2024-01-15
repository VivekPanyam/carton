namespace Carton.Models.Tensors;

public class IntTensor : GenericTensor<int>
{
    public IntTensor(string key, params ulong[] shape) : base(key, shape)
    {
    }

    public IntTensor(string key, int value) : base(key, 1)
    {
        Values[0] = value;
    }
}