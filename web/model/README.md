# web/model/

The page's in-browser solver (*Neural network (ONNX, in the browser)*) loads `policy-v2.onnx`, the ONNX export of
the trained policy network (its contract is in [`models/README.md`](../../models/README.md)). It tries these URLs,
relative to the page, and uses the first one that loads:

1. `../models/policy-v2.onnx`: the whole repository is served and the page is `/web/index.html`, for example
   `python -m rubiks.serve --web .` and then `http://localhost:8765/web/`.
2. `models/policy-v2.onnx`: a site that publishes the page with a `models/` folder next to `index.html`.
3. `model/policy-v2.onnx`: this folder. The GitHub Pages site publishes `web/` with the model copied here from
   `models/` (see `.github/workflows/pages.yml`).

`python -m rubiks.serve --web web` serves `web/` only, so with it the model has to be a real file in this folder:

```
mkdir -p web/model && cp models/policy-v2.onnx web/model/
```

A symbolic link does not work, because the server refuses every file that resolves outside `web/`. The repository
keeps the model only in `models/`, not here. The **Model URL** field under the backend replaces the lookup with one
URL of your choice.

Under `file://` the browser cannot fetch the model at all; the page then says that the backend needs a server or the
GitHub Pages URL.
