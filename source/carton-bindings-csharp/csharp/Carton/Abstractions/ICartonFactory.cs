using Carton.Models;

namespace Carton;

public interface ICartonFactory : IDisposable
{
    /// <summary>
    /// Loads a Carton model from an url or a path asynchonous.
    /// </summary>
    /// <param name="urlOrPath">String containing url or path</param>
    /// <returns></returns>
    Task<CartonModel> LoadModelAsync(string urlOrPath);
}
