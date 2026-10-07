# Restoring a version of the workshop

A version ties together the code, the notebooks, the sheets and the models. `RELEASE.json` identifies the version, the expected models-and-videos archive, and the checksums of every shipped file.

## Restore a published version

In a fresh clone of the repository:

```bash
git fetch --tags
git tag --list
git switch --detach v1.1.0
./scripts/fetch_assets.sh
```

Choose the tag you want from the list. The download uses the tag written in the manifest of this checkout, never implicitly the latest release. A different or altered archive is refused before anything in the runtime is changed.

To work offline, keep the repository and the downloaded archive of that same release:

```bash
ASSETS_FILE=/path/workshop-assets-v1.1.0.tar.gz ./scripts/fetch_assets.sh
```

## Create a variant

Create a branch from the chosen tag. Keep the main workshop working, together with its material. A hardware extension gets its own version and its own validation results; validating the simulation workshop does not prove that a physical arm works.

## Retention rules

A published release is never replaced or renamed. Every fix gets a new number, even if only the material changes. Tags and archives stay available. A version containing `-rc` is a candidate: it does not mean that the teaching rehearsal was validated.

Going back in the code is not enough to restore a machine. Also keep the image identifiers per region, the deployment settings, the software versions and the test reports. Test an old image on a new machine before replacing an environment in service. Personal credentials and keys must never be stored in the repository.
