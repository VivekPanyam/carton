namespace Carton.Helpers;

using static Native.CartonBindings;

public static class DataTypeHelper
{
    /// <summary>
    ///     Gets the Carton DataType for a specific Type.
    /// </summary>
    /// <param name="type">The type to map.</param>
    /// <returns></returns>
    public static DataType GetDataTypeForType(Type type)
    {
        if (type == typeof(float))
        {
            return DataType.DATA_TYPE_FLOAT;
        }

        if (type == typeof(double))
        {
            return DataType.DATA_TYPE_DOUBLE;
        }

        if (type == typeof(string))
        {
            return DataType.DATA_TYPE_STRING;
        }

        if (type == typeof(short))
        {
            return DataType.DATA_TYPE_I16;
        }

        if (type == typeof(int))
        {
            return DataType.DATA_TYPE_I32;
        }

        if (type == typeof(long))
        {
            return DataType.DATA_TYPE_I64;
        }

        if (type == typeof(ushort))
        {
            return DataType.DATA_TYPE_U16;
        }

        if (type == typeof(uint))
        {
            return DataType.DATA_TYPE_U32;
        }

        if (type == typeof(ulong))
        {
            return DataType.DATA_TYPE_U64;
        }

        return DataType.DATA_TYPE_U32;
    }
}