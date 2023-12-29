These bindings will allow users to use Carton within C#.

The C# bindings are generated from Cartons' C bindings. (`carton-bindings-c`)

The project `Carton.Tests` can be referenced to see some examples of the usage.

## Details on generating bindings

### Dependencies

For generating the bindings we use the dotnet tool [c2cs](https://github.com/bottlenoselabs/c2cs/).
Install the these tools for generating the bindings:
```
$ dotnet tool install bottlenoselabs.c2cs.tool --global 
$ dotnet tool install bottlenoselabs.castffi.tool -g
```

### Preparing c2cs

`c2cs` needs a little setup before it can generate the bindings correctly.
We will need to generate some config files.
These are `config-windows.json`, `config-linux.json` & `config-macos.json`. I have not yet been able to create the last one.
We then use the `castffi extract` command to extract the abstract syntax tree (ast).
```
$ castffi extract config-windows.json
$ castffi extract config-linux.json
$ castffi extract config-macos.json
```

We can then merge these seperate ast-files into a single cross platform ast using:
`castffi merge --inputDirectoryPath ./ast --outputFilePath cross-platform-ast.json`

Lastly we create the `c2cs` config file named `config-generate-cs.json`.

### Running c2cs

To run the tool we simply execute the following command:
`c2cs generate --config config-generate-cs.json`

This will have regenerated the binding file which you can find in `csharp/Carton/CartonBindings.gen.cs`.