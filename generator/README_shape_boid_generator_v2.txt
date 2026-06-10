Build:
  g++ -O2 -std=c++17 -o shape_boid_generator_v2 shape_boid_generator_v2.cpp

Run:
  ./shape_boid_generator_v2 example_simple.boid
  ./shape_boid_generator_v2 example_simple.boid output.grid

Supported shapes:
  - box
  - sphere
  - cylinder
  - ring2d
  - triangle_prism

Supported operation:
  - Mode add
  - Mode subtract

Notes:
  - All shapes use one global ParticleDistance, which keeps the format simple.
  - RotateDeg is Euler XYZ in degrees.
  - triangle_prism uses local XY points P1/P2/P3, then extrudes by Height in local Z.
