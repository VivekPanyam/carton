namespace Carton.Tests;

using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Models.Tensors;

[TestClass]
public class CartonFactoryTests
{
    [TestMethod]
    public async Task LoadAndInfer_BertBaseUncased_Test()
    {
        using ICartonFactory cartonFactory = new CartonFactory();
        var cartonModel = await cartonFactory.LoadModelAsync("https://carton.pub/google-research/bert-base-uncased/5f26d87c5d82b7c37ebf92fcb38788a063d49a64cfcf1f9d118b3b710bb88005");

        Assert.IsNotNull(cartonModel);

        var result = await cartonModel.Infer(new ITensor[] { new StringTensor("input", "Today is a good [MASK].") });
        var tokens = await result.GetScalarString("tokens");
        Assert.AreEqual("day", tokens);

        var scores = await result.GetScalarFloat("scores");
        Assert.AreEqual(14.551313400268555, scores);
    }

    [TestMethod]
    public async Task LoadAndInfer_BertBaseUncased_MultipleTensors_Test()
    {
        using ICartonFactory cartonFactory = new CartonFactory();
        var cartonModel = await cartonFactory.LoadModelAsync("https://carton.pub/google-research/bert-base-uncased/5f26d87c5d82b7c37ebf92fcb38788a063d49a64cfcf1f9d118b3b710bb88005");

        Assert.IsNotNull(cartonModel);

        var result = await cartonModel.Infer(new ITensor[] { new StringTensor("input", "Today is a good [MASK]."), new IntTensor("max_tokens", 5) });
        var tokens = await result.GetScalarString("tokens");
        Assert.AreEqual("day", tokens);

        var scores = await result.GetScalarFloat("scores");
        Assert.AreEqual(14.551313400268555, scores);
    }

    [TestMethod]
    public async Task LoadAndInfer_BertBaseUncased_NonScalar_Test()
    {
        using ICartonFactory cartonFactory = new CartonFactory();
        var cartonModel = await cartonFactory.LoadModelAsync("https://carton.pub/google-research/bert-base-uncased/5f26d87c5d82b7c37ebf92fcb38788a063d49a64cfcf1f9d118b3b710bb88005");

        Assert.IsNotNull(cartonModel);

        var input = new StringTensor("input", 2);
        input.LoadFromArray(new[] { "Paris is the [MASK] of France.", "Today is a good [MASK]." });

        var result = await cartonModel.Infer(new ITensor[] { input });
        var tokens = await result.GetStringTensor("tokens", new ulong[] { 2, 1 });

        Assert.AreEqual("capital", tokens[0]);
        Assert.AreEqual("day", tokens[1]);

        var scores = await result.GetTensor<float>("scores", new ulong[] { 2, 1 });
        Assert.AreEqual(18.19973373413086, scores[0]);
        Assert.AreEqual(12.977392196655273, scores[1]);
    }

    [TestMethod]
    public async Task LoadAndInfer_BertLargeCNN_Test()
    {
        using ICartonFactory cartonFactory = new CartonFactory();
        var cartonModel = await cartonFactory.LoadModelAsync("https://carton.pub/facebook/bart-large-cnn/bbc5eab187eeab1cad5cf4ff3248ddd9a25fed27d82707de8b6fe85976007330");

        Assert.IsNotNull(cartonModel);

        var result = await cartonModel.Infer(new[]
                                             {
                                                 new StringTensor("input", """
                                                                           NASA’s James Webb Space Telescope has followed up on observations by the Hubble Space Telescope of the farthest star ever detected in the very distant universe, within the first billion years after the big bang. Webb’s NIRCam (Near-Infrared Camera) instrument reveals the star to be a massive B-type star more than twice as hot as our Sun, and about a million times more luminous.

                                                                           The star, which the research team has dubbed Earendel, is located in the Sunrise Arc galaxy and is detectable only due to the combined power of human technology and nature via an effect called gravitational lensing. Both Hubble and Webb were able to detect Earendel due to its lucky alignment behind a wrinkle in space-time created by the massive galaxy cluster WHL0137-08. The galaxy cluster, located between us and Earendel, is so massive that it warps the fabric of space itself, which produces a magnifying effect, allowing astronomers to look through the cluster like a magnifying glass.

                                                                           While other features in the galaxy appear multiple times due to the gravitational lensing, Earendel only appears as a single point of light even in Webb’s high-resolution infrared imaging. Based on this, astronomers determine the object is magnified by a factor of at least 4,000, and thus is extremely small – the most distant star ever detected, observed 1 billion years after the big bang. The previous record-holder for the most distant star was detected by Hubble and observed around 4 billion years after the big bang. Another research team using Webb recently identified a gravitationally lensed star they nicknamed Quyllur, a red giant star observed 3 billion years after the big bang.

                                                                           Stars as massive as Earendel often have companions. Astronomers did not expect Webb to reveal any companions of Earendel since they would be so close together and indistinguishable on the sky. However, based solely on the colors of Earendel, astronomers think they see hints of a cooler, redder companion star. This light has been stretched by the expansion of the universe to wavelengths longer than Hubble’s instruments can detect, and so was only detectable with Webb.

                                                                           Webb’s NIRCam also shows other notable details in the Sunrise Arc, which is the most highly magnified galaxy yet detected in the universe’s first billion years. Features include both young star-forming regions and older established star clusters as small as 10 light-years across. On either side of the wrinkle of maximum magnification, which runs right through Earendel, these features are mirrored by the distortion of the gravitational lens. The region forming stars appears elongated, and is estimated to be less than 5 million years old. Smaller dots on either side of Earendel are two images of one older, more established star cluster, estimated to be at least 10 million years old. Astronomers determined this star cluster is gravitationally bound and likely to persist until the present day. This shows us how the globular clusters in our own Milky Way might have looked when they formed 13 billion years ago.

                                                                           Astronomers are currently analyzing data from Webb’s NIRSpec (Near-Infrared Spectrograph) instrument observations of the Sunrise Arc galaxy and Earendel, which will provide precise composition and distance measurements for the galaxy.

                                                                           Since Hubble’s discovery of Earendel, Webb has detected other very distant stars using this technique, though none quite as far as Earendel. The discoveries have opened a new realm of the universe to stellar physics, and new subject matter to scientists studying the early universe, where once galaxies were the smallest detectable cosmic objects. The research team has cautious hope that this could be a step toward the eventual detection of one of the very first generation of stars, composed only of the raw ingredients of the universe created in the big bang – hydrogen and helium.
                                                                           """)
                                             });
        var output = await result.GetScalarString("output");

        Assert.AreEqual("The star, which the research team has dubbed Earendel, is located in the Sunrise Arc galaxy and is detectable only due to the combined power of human technology and nature via an effect called gravitational lensing. The star is a massive B-type star more than twice as hot as our Sun, and about a million times more luminous.", output);
    }
}