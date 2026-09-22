# Gaussian splat to textured mesh

Filter, densify and Poisson-mesh a trained 3D Gaussian splat, then bake a real texture. CPU only.

**Get the dataset:** [learngeodata.eu/materials/gaussian-splat-to-mesh/](https://learngeodata.eu/materials/gaussian-splat-to-mesh/)
**More tutorials:** [medium.com/@florentpoux](https://medium.com/@florentpoux)


---

## What is here

- `splat_to_textured_mesh.py` The nine-function pipeline, start to finish.

## Run it

```bash
pip install -r requirements.txt
python splat_to_textured_mesh.py
```

Download the scene from the [kit page](https://learngeodata.eu/materials/gaussian-splat-to-mesh/), put
it next to the script, and run. The script prints its numbers as it goes so you
can check them against the article at every step.

Requires Python 3.10, numpy, scipy, plyfile, open3d, trimesh, xatlas, pillow.

## What you get

- `cabin_in_the_forest.ply` The trained splat, 84,528 Gaussians, 20 MB.
- `cabin_textured.glb` The finished mesh, so you can check your result against mine.

## License

Code: MIT, see [LICENSE](../LICENSE). The dataset is distributed from the kit
page under its own terms and is not covered by this repository's license.

## Author

Dr. Florent Poux, 3D Geodata Academy.
[learngeodata.eu](https://learngeodata.eu) · [ORCID 0000-0001-6368-4399](https://orcid.org/0000-0001-6368-4399)

Author of [3D Data Science with Python](https://www.oreilly.com/library/view/3d-data-science/9781098161323/) (O'Reilly Media, 2025).
