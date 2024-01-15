using static Carton.Native.CartonBindings;

namespace Carton.Models.Tensors;

using bottlenoselabs.C2CS.Runtime;

public class StringTensor : GenericTensor<string>
{
    public StringTensor(string key, params ulong[] shape) : base(key, shape)
    {
    }

    public StringTensor(string key, string value) : base(key, 1)
    {
        Values[0] = value;
    }

    /// <inheritdoc />
    public override unsafe CartonTensor* ToCartonTensor()
    {
        if (Shape == new ulong[] { 1 })
        {
            return ToScalarTensor();
        }

        return ToTensor();
    }

    private unsafe CartonTensor* ToTensor()
    {
        CartonTensor* tensor;

        fixed (ulong* shapePtr = Shape)
        {
            carton_tensor_create(DataType.DATA_TYPE_STRING, shapePtr, (ulong)Shape.Length, &tensor);
        }
        
        for (ulong i = 0; i < (ulong)Values.Length; i++)
        {
            carton_tensor_set_string(tensor, i, CString.FromString(Values[i]));
        }

        return tensor;
    }

    private unsafe CartonTensor* ToScalarTensor()
    {
        CartonTensor* tensor;
        ulong shape = 1;
        ulong dims = 1;
        carton_tensor_create(DataType.DATA_TYPE_STRING, &shape, dims, &tensor);

        for (ulong i = 0; i < Elements; i++)
        {
            carton_tensor_set_string(tensor, i, CString.FromString(Values[i]));
        }

        return tensor;
    }
}