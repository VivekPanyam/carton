**The contents of this folder directly affect the wire protocol and should generally NOT be modified**

If you are modifying code in this folder, you have two options:

1. Bump the interface version (and the major version of this crate). Usually only for large, well planned changes.

    - This can lead to bloat of the main library because it builds against *every* version of the runner interface.

2. Make a compatible change. Be careful.

Any changes must be compatible in the following situations

For an example major version `2`:


Client version `2.x.y` <----> Runner version `2.x.y` for any `x` and `y`.

So backwards and forwards compatibility:

- What happens if an old client talks to a runner built against the new version of the wire protocol?
- What happens if a new client talks to a runner built against and old version of the wire protocol?