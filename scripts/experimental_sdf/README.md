# Four-hole SDF experiment

This is a **benchmark-only**, fixed-dimension, uncalibrated model. It is not a
robot connector, does not simulate magnetic attraction, and must not be used
for clearance checking, safety validation, planning, or control. The `max` CSG
field is not an exact signed-distance field at shape intersections.

Run from the repository root with the existing shared MuJoCo environment:

```bash
./scripts/python.sh -m scripts.bench_connector_sdf
```

The command compiles the experimental plugin into a temporary directory using
the `c++` compiler and the headers/library from the shared `mujoco` package;
it installs or changes no dependency. It compares isolated two-free-body
geometry for 1 N axial compression, no contact and a side-offset case. The
benchmark times both the physics step and contact-statistics collection, so
absolute values cannot be compared to timings without those diagnostics.

The four-hole disc uses radius 50 mm, thickness 2 mm, holes of radius 2.1 mm
at pitch radius 35 mm, and pins of radius 2 mm and height 2 mm. These are
hypothetical values. The non-exact CSG field and MuJoCo's iterative SDF
collision search can produce incorrect contact normals, depths, or missing
contacts; results are not representative of every SDF implementation.

In MuJoCo 3.11.0 with 40 starting points and 10 iterations, the aligned pin
has an analytical 0.1 mm gap, yet `mj_geomDistance` returned the 10 mm query
cutoff. A 0.3 mm sideways offset has an approximately 0.2 mm geometric
overlap, while the deepest returned contact was approximately 0.094 mm for
each of the four pins. These observations disqualify this prototype from the
existing MPC safety clearance checks until independently fixed and validated.
