using System.Threading.Tasks;
using Carton.Models.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Carton.Tests
{
    [TestClass]
    public class CartonFactoryTests
    {
        [TestMethod]
        public async Task LoadAndInfer_Test()
        {
            using (ICartonFactory cartonFactory = new CartonFactory())
            {
                var cartonModel = await cartonFactory.LoadModelAsync("https://carton.pub/google-research/bert-base-uncased/5f26d87c5d82b7c37ebf92fcb38788a063d49a64cfcf1f9d118b3b710bb88005");

                Assert.IsNotNull(cartonModel);

                var result = await cartonModel.Infer(new[] { new StringTensor("input", "Today is a good [MASK].") });
                var tokens = await result.GetAndRemoveString("tokens");
                var scores = await result.GetAndRemoveFloat("scores");

                Assert.AreEqual("day", tokens);
                Assert.AreEqual(14.551313400268555, scores);
            }
        }
    }
}
