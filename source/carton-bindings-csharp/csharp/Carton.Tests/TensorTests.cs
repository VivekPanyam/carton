namespace Carton.Tests;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Models.Tensors;

[TestClass]
public class TensorTests
{
    [TestMethod]
    public void Tensor1D()
    {
        var values = new[] { 1, 2, 3, 4, 5, 6, 7 };
        var tensor1D = new IntTensor("Test1D", 7);
        tensor1D.LoadFromArray(values);

        Assert.AreEqual("Test1D", tensor1D.Key);
        Assert.AreEqual(7, tensor1D.Elements);

        Assert.AreEqual(1, tensor1D[0]);
        Assert.AreEqual(2, tensor1D[1]);
        Assert.AreEqual(3, tensor1D[2]);
        Assert.AreEqual(4, tensor1D[3]);
        Assert.AreEqual(5, tensor1D[4]);
        Assert.AreEqual(6, tensor1D[5]);
        Assert.AreEqual(7, tensor1D[6]);

        Assert.AreEqual(values, tensor1D.Raw);
    }

    [TestMethod]
    public void Tensor2D()
    {
        var values = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        var tensor2D = new IntTensor("Test2D", 3, 5);
        tensor2D.LoadFromArray(values);

        Assert.AreEqual("Test2D", tensor2D.Key);
        Assert.AreEqual(15, tensor2D.Elements);

        Assert.AreEqual(1, tensor2D[0, 0]);
        Assert.AreEqual(2, tensor2D[0, 1]);
        Assert.AreEqual(3, tensor2D[0, 2]);
        Assert.AreEqual(4, tensor2D[0, 3]);
        Assert.AreEqual(5, tensor2D[0, 4]);

        Assert.AreEqual(6, tensor2D[1, 0]);
        Assert.AreEqual(7, tensor2D[1, 1]);
        Assert.AreEqual(8, tensor2D[1, 2]);
        Assert.AreEqual(9, tensor2D[1, 3]);
        Assert.AreEqual(10, tensor2D[1, 4]);

        Assert.AreEqual(11, tensor2D[2, 0]);
        Assert.AreEqual(12, tensor2D[2, 1]);
        Assert.AreEqual(13, tensor2D[2, 2]);
        Assert.AreEqual(14, tensor2D[2, 3]);
        Assert.AreEqual(15, tensor2D[2, 4]);

        Assert.AreEqual(values, tensor2D.Raw);
    }


    [TestMethod]
    public void Tensor3D()
    {
        var values = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
        var tensor3D = new IntTensor("Test3D", 3, 3, 2);
        tensor3D.LoadFromArray(values);

        Assert.AreEqual("Test3D", tensor3D.Key);
        Assert.AreEqual(18, tensor3D.Elements);

        Assert.AreEqual(1, tensor3D[0, 0, 0]);
        Assert.AreEqual(2, tensor3D[0, 0, 1]);

        Assert.AreEqual(3, tensor3D[0, 1, 0]);
        Assert.AreEqual(4, tensor3D[0, 1, 1]);

        Assert.AreEqual(5, tensor3D[0, 2, 0]);
        Assert.AreEqual(6, tensor3D[0, 2, 1]);

        Assert.AreEqual(7, tensor3D[1, 0, 0]);
        Assert.AreEqual(8, tensor3D[1, 0, 1]);

        Assert.AreEqual(9, tensor3D[1, 1, 0]);
        Assert.AreEqual(10, tensor3D[1, 1, 1]);

        Assert.AreEqual(11, tensor3D[1, 2, 0]);
        Assert.AreEqual(12, tensor3D[1, 2, 1]);

        Assert.AreEqual(13, tensor3D[2, 0, 0]);
        Assert.AreEqual(14, tensor3D[2, 0, 1]);

        Assert.AreEqual(15, tensor3D[2, 1, 0]);
        Assert.AreEqual(16, tensor3D[2, 1, 1]);

        Assert.AreEqual(17, tensor3D[2, 2, 0]);
        Assert.AreEqual(18, tensor3D[2, 2, 1]);

        Assert.AreEqual(values, tensor3D.Raw);
    }
}