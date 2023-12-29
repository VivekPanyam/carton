using bottlenoselabs.C2CS.Runtime;
using static Carton.Native.CartonBindings;

namespace Carton.Models.Tensors;

public class StringTensor : ITensor
{
    private string Key;
    public string Value;

    public StringTensor(string key, string value)
    {
        Key = key;
        Value = value;
    }

    /// <inheritdoc/>
    public string GetKey() => Key;

    /// <inheritdoc/>
    public unsafe CartonTensor* ToCartonTensor()
    {
        CartonTensor* tensor;
        ulong shape = 1;
        ulong dims = 1;
        carton_tensor_create(DataType.DATA_TYPE_STRING, &shape, dims, &tensor);
        carton_tensor_set_string(tensor, 0, CString.FromString(Value));

        return tensor;
    }
}
