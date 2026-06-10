#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Vec2 {
    double x = 0.0, y = 0.0;
};

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3 operator+(const Vec3& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }

    Vec3 operator-(const Vec3& o) const {
        return {x - o.x, y - o.y, z - o.z};
    }

    Vec3 operator*(double s) const {
        return {x * s, y * s, z * s};
    }
};

struct Mat3 {
    double a[3][3]{};
};

static inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static inline std::string remove_comment(const std::string& s) {
    size_t p = s.find('#');
    return (p == std::string::npos) ? s : s.substr(0, p);
}

static inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

static Mat3 make_mat3(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22
) {
    Mat3 M{};
    M.a[0][0] = a00; M.a[0][1] = a01; M.a[0][2] = a02;
    M.a[1][0] = a10; M.a[1][1] = a11; M.a[1][2] = a12;
    M.a[2][0] = a20; M.a[2][1] = a21; M.a[2][2] = a22;
    return M;
}

static Mat3 matmul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                C.a[i][j] += A.a[i][k] * B.a[k][j];
            }
        }
    }
    return C;
}

static Vec3 mul(const Mat3& A, const Vec3& v) {
    return {
        A.a[0][0] * v.x + A.a[0][1] * v.y + A.a[0][2] * v.z,
        A.a[1][0] * v.x + A.a[1][1] * v.y + A.a[1][2] * v.z,
        A.a[2][0] * v.x + A.a[2][1] * v.y + A.a[2][2] * v.z
    };
}

static Mat3 transpose(const Mat3& A) {
    Mat3 T{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T.a[i][j] = A.a[j][i];
        }
    }
    return T;
}

static Mat3 rotation_xyz_deg(const Vec3& deg) {
    double rx = deg2rad(deg.x);
    double ry = deg2rad(deg.y);
    double rz = deg2rad(deg.z);

    Mat3 Rx = make_mat3(
        1, 0, 0,
        0, std::cos(rx), -std::sin(rx),
        0, std::sin(rx),  std::cos(rx)
    );

    Mat3 Ry = make_mat3(
         std::cos(ry), 0, std::sin(ry),
         0,            1, 0,
        -std::sin(ry), 0, std::cos(ry)
    );

    Mat3 Rz = make_mat3(
        std::cos(rz), -std::sin(rz), 0,
        std::sin(rz),  std::cos(rz), 0,
        0, 0, 1
    );

    return matmul(Rz, matmul(Ry, Rx));
}

enum class ShapeKind {
    BOX,
    SPHERE,
    CYLINDER,
    RING2D,
    TRIANGLE,
    TRIANGLE_PRISM
};

enum class ShapeMode {
    ADD,
    SUBTRACT
};

struct Range1D {
    bool has = false;
    double lo = 0.0;
    double hi = 0.0;
};

struct Shape {
    std::string name = "shape";
    ShapeKind kind = ShapeKind::BOX;
    ShapeMode mode = ShapeMode::ADD;

    int type = 0;
    int rigidType = -1;

    Vec3 velocity{};
    double enthalpy = 0.0;
    Vec3 rotateDeg{};

    bool hasSpacing = false;
    Vec3 spacing{};

    // Pivot used for rotation.
    // By default, cuboids are rotated around the maximum-x and maximum-y corner:
    // (Upper.x, Upper.y, 0.5*(Lower.z+Upper.z)).
    bool hasRotationPivot = false;
    Vec3 rotationPivot{};

    Range1D xr, yr, zr;

    bool zSingleLayer = false;
    double zLayerValue = 0.0;

    bool hasCenter = false;
    bool hasSize = false;
    Vec3 center{};
    Vec3 size{};

    double radius = 0.0;
    double innerRadius = 0.0;
    double height = 0.0;
    double ratio = 0.0;

    Vec2 p1{}, p2{}, p3{};
    bool hasTriangle = false;
};

struct Particle {
    int type = 0;
    int rigidType = -1;
    Vec3 pos{};
    Vec3 vel{};
    double enthalpy = 0.0;
};

struct Scene {
    int dimension = 3;
    double particleDistance = -1.0;

    Vec3 lowerDomain{};
    Vec3 upperDomain{};

    bool lowerSet = false;
    bool upperSet = false;

    std::vector<Shape> shapes;
};

static ShapeKind parse_kind(const std::string& s) {
    if (s == "box") return ShapeKind::BOX;
    if (s == "sphere") return ShapeKind::SPHERE;
    if (s == "cylinder") return ShapeKind::CYLINDER;
    if (s == "ring2d") return ShapeKind::RING2D;
    if (s == "triangle") return ShapeKind::TRIANGLE;
    if (s == "triangle_prism") return ShapeKind::TRIANGLE_PRISM;

    throw std::runtime_error("Unknown Kind: " + s);
}

static ShapeMode parse_mode(const std::string& s) {
    if (s == "add") return ShapeMode::ADD;
    if (s == "subtract") return ShapeMode::SUBTRACT;

    throw std::runtime_error("Unknown Mode: " + s);
}

static bool parse_vec2(std::istringstream& iss, Vec2& v) {
    return bool(iss >> v.x >> v.y);
}

static bool parse_vec3(std::istringstream& iss, Vec3& v) {
    return bool(iss >> v.x >> v.y >> v.z);
}

static bool parse_vec3_optional_z(std::istringstream& iss, Vec3& v, double defaultZ) {
    if (!(iss >> v.x >> v.y)) return false;
    if (!(iss >> v.z)) v.z = defaultZ;
    return true;
}

static void normalize_range(Range1D& r) {
    if (r.has && r.lo > r.hi) std::swap(r.lo, r.hi);
}

static bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        char ca = (char)std::tolower((unsigned char)a[i]);
        char cb = (char)std::tolower((unsigned char)b[i]);
        if (ca != cb) return false;
    }
    return true;
}

static void finalize_shape(Shape& s, double dx, int dimension) {
    if (dimension == 2 && !s.zr.has && !s.zSingleLayer) {
        s.zr = {true, 0.0, dx};
    }

    if (s.zSingleLayer && !s.zr.has) {
        s.zr = {true, s.zLayerValue - 0.5 * dx, s.zLayerValue + 0.5 * dx};
    }

    normalize_range(s.xr);
    normalize_range(s.yr);
    normalize_range(s.zr);

    if (s.kind == ShapeKind::BOX && (!s.xr.has || !s.yr.has || !s.zr.has)) {
        if (!s.hasCenter || !s.hasSize) {
            throw std::runtime_error("box requires XRange/YRange/ZRange or Center+Size");
        }

        s.xr = {true, s.center.x - 0.5 * s.size.x, s.center.x + 0.5 * s.size.x};
        s.yr = {true, s.center.y - 0.5 * s.size.y, s.center.y + 0.5 * s.size.y};
        s.zr = {true, s.center.z - 0.5 * s.size.z, s.center.z + 0.5 * s.size.z};
    }

    if (!s.xr.has || !s.yr.has || !s.zr.has) {
        if (s.hasCenter && s.hasSize) {
            s.xr = {true, s.center.x - 0.5 * s.size.x, s.center.x + 0.5 * s.size.x};
            s.yr = {true, s.center.y - 0.5 * s.size.y, s.center.y + 0.5 * s.size.y};
            s.zr = {true, s.center.z - 0.5 * s.size.z, s.center.z + 0.5 * s.size.z};
        } else {
            throw std::runtime_error("shape requires XRange/YRange/ZRange or Center+Size");
        }
    }

    if (!s.hasCenter) {
        s.center.x = 0.5 * (s.xr.lo + s.xr.hi);
        s.center.y = 0.5 * (s.yr.lo + s.yr.hi);
        s.center.z = 0.5 * (s.zr.lo + s.zr.hi);
        s.hasCenter = true;
    }

    // Default rotation pivot requested for original cuboid format:
    // not the gravity/geometric center, but the maximum x and maximum y corner.
    // For 2D cases, z is kept at the middle of the thin layer.
    if (!s.hasRotationPivot) {
        s.rotationPivot.x = s.xr.hi;
        s.rotationPivot.y = s.yr.hi;
        s.rotationPivot.z = 0.5 * (s.zr.lo + s.zr.hi);
        s.hasRotationPivot = true;
    }

    if (s.kind == ShapeKind::TRIANGLE || s.kind == ShapeKind::TRIANGLE_PRISM) {
        if (!s.hasTriangle) {
            throw std::runtime_error("triangle requires P1/P2/P3");
        }
    }

    if (s.kind == ShapeKind::CYLINDER || s.kind == ShapeKind::RING2D) {
        if (s.radius <= 0.0) {
            s.radius = 0.5 * (s.xr.hi - s.xr.lo);
        }
    }

    if (s.kind == ShapeKind::RING2D) {
        if (s.innerRadius <= 0.0 && s.ratio > 0.0) {
            s.innerRadius = s.radius * s.ratio;
        }
    }

    if (s.kind == ShapeKind::SPHERE) {
        if (s.radius <= 0.0) {
            s.radius = 0.5 * (s.xr.hi - s.xr.lo);
        }

        if (s.innerRadius <= 0.0 && s.ratio > 0.0) {
            s.innerRadius = s.radius * s.ratio;
        }
    }

    if (s.height <= 0.0) {
        s.height = s.zr.hi - s.zr.lo;
    }
}

static Scene read_scene(const std::string& filename) {
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    Scene scene;
    std::string raw;
    int lineNo = 0;
    bool inShape = false;
    Shape cur{};

    auto begin_shape = [&](const std::string& key, std::istringstream& iss) {
        if (inShape) {
            throw std::runtime_error("Nested shape/cuboid block at line " + std::to_string(lineNo));
        }
        inShape = true;
        cur = Shape{};

        // New format: Shape ... EndShape. The kind is set by Kind.
        // Original/convenient format: StartCuboid ... EndCuboid.
        if (iequals(key, "StartCuboid") || iequals(key, "Cuboid")) {
            cur.kind = ShapeKind::BOX;
            cur.name = "cuboid";
        } else if (iequals(key, "StartSphere")) {
            cur.kind = ShapeKind::SPHERE;
            cur.name = "sphere";
        } else if (iequals(key, "StartCylinder")) {
            cur.kind = ShapeKind::CYLINDER;
            cur.name = "cylinder";
        } else if (iequals(key, "StartRing2D")) {
            cur.kind = ShapeKind::RING2D;
            cur.name = "ring2d";
        } else if (iequals(key, "StartTriangle")) {
            cur.kind = ShapeKind::TRIANGLE;
            cur.name = "triangle";
        } else if (iequals(key, "StartTrianglePrism")) {
            cur.kind = ShapeKind::TRIANGLE_PRISM;
            cur.name = "triangle_prism";
        }

        std::string maybeName;
        if (iss >> maybeName) {
            cur.name = maybeName;
        }
    };

    auto end_shape = [&]() {
        if (!inShape) {
            throw std::runtime_error("End block without a matching Start block at line " + std::to_string(lineNo));
        }

        finalize_shape(cur, scene.particleDistance > 0.0 ? scene.particleDistance : 1.0,
                       scene.dimension);
        scene.shapes.push_back(cur);
        inShape = false;
    };

    while (std::getline(ifs, raw)) {
        ++lineNo;

        std::string line = trim(remove_comment(raw));
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (iequals(key, "Shape") || iequals(key, "StartShape") ||
            iequals(key, "StartCuboid") || iequals(key, "Cuboid") ||
            iequals(key, "StartSphere") || iequals(key, "StartCylinder") ||
            iequals(key, "StartRing2D") || iequals(key, "StartTriangle") ||
            iequals(key, "StartTrianglePrism")) {
            begin_shape(key, iss);
            continue;
        }

        if (iequals(key, "EndShape") || iequals(key, "EndCuboid") ||
            iequals(key, "EndSphere") || iequals(key, "EndCylinder") ||
            iequals(key, "EndRing2D") || iequals(key, "EndTriangle") ||
            iequals(key, "EndTrianglePrism")) {
            end_shape();
            continue;
        }

        if (!inShape) {
            if (key == "Dimension") {
                iss >> scene.dimension;
                if (scene.dimension != 2 && scene.dimension != 3) {
                    throw std::runtime_error("Dimension must be 2 or 3");
                }
            } else if (key == "ParticleDistance" || key == "ParticleSpacing") {
                iss >> scene.particleDistance;
            } else if (key == "LowerDomain") {
                if (!parse_vec3_optional_z(iss, scene.lowerDomain, 0.0)) {
                    throw std::runtime_error("Bad LowerDomain at line " + std::to_string(lineNo));
                }
                scene.lowerSet = true;
            } else if (key == "UpperDomain") {
                double defaultZ = scene.particleDistance > 0.0 ? scene.particleDistance : 1.0;
                if (!parse_vec3_optional_z(iss, scene.upperDomain, defaultZ)) {
                    throw std::runtime_error("Bad UpperDomain at line " + std::to_string(lineNo));
                }
                scene.upperSet = true;
            } else if (key == "Domain") {
                iss >> scene.lowerDomain.x >> scene.lowerDomain.y >> scene.lowerDomain.z
                    >> scene.upperDomain.x >> scene.upperDomain.y >> scene.upperDomain.z;
                scene.lowerSet = true;
                scene.upperSet = true;
            } else if (key == "DomainX") {
                iss >> scene.lowerDomain.x >> scene.upperDomain.x;
                scene.lowerSet = true;
                scene.upperSet = true;
            } else if (key == "DomainY") {
                iss >> scene.lowerDomain.y >> scene.upperDomain.y;
                scene.lowerSet = true;
                scene.upperSet = true;
            } else if (key == "DomainZ") {
                std::string a;
                iss >> a;

                if (a == "auto" || a == "Auto" || a == "AUTO") {
                    scene.lowerDomain.z = 0.0;
                    scene.upperDomain.z = scene.particleDistance;
                } else {
                    scene.lowerDomain.z = std::stod(a);
                    iss >> scene.upperDomain.z;
                }

                scene.lowerSet = true;
                scene.upperSet = true;
            } else {
                throw std::runtime_error("Unknown global key at line " +
                                         std::to_string(lineNo) + ": " + key);
            }

            continue;
        }

        if (key == "Name") {
            iss >> cur.name;
        } else if (key == "Kind") {
            std::string s;
            iss >> s;
            cur.kind = parse_kind(s);
        } else if (key == "Mode") {
            std::string s;
            iss >> s;
            cur.mode = parse_mode(s);
        } else if (key == "Type") {
            iss >> cur.type;
        } else if (key == "RigidType") {
            iss >> cur.rigidType;
        } else if (key == "Velocity") {
            if (!parse_vec3_optional_z(iss, cur.velocity, 0.0)) {
                throw std::runtime_error("Bad Velocity at line " + std::to_string(lineNo));
            }
        } else if (key == "Enthalpy") {
            iss >> cur.enthalpy;
        } else if (key == "RotateDeg" || key == "RotationDeg") {
            if (!parse_vec3_optional_z(iss, cur.rotateDeg, 0.0)) {
                throw std::runtime_error("Bad RotateDeg at line " + std::to_string(lineNo));
            }
        } else if (key == "AngleDeg" || key == "ThetaDeg") {
            // Convenient 2D notation: rotate around z-axis.
            iss >> cur.rotateDeg.z;
        } else if (key == "Spacing" || key == "ParticleDistance" || key == "ParticleSpacing") {
            // For compatibility with original cuboid blocks.
            // One value: uniform spacing. Two values: x/y with z=global spacing. Three values: x/y/z.
            double a = 0.0, b = 0.0, c = 0.0;
            if (!(iss >> a)) {
                throw std::runtime_error("Bad Spacing at line " + std::to_string(lineNo));
            }
            if (!(iss >> b)) b = a;
            if (!(iss >> c)) c = (scene.particleDistance > 0.0 ? scene.particleDistance : a);
            cur.spacing = {a, b, c};
            cur.hasSpacing = true;
        } else if (key == "Center" || key == "GravityCenter") {
            double defaultZ = scene.dimension == 2 ? 0.5 * (scene.particleDistance > 0.0 ? scene.particleDistance : 1.0) : 0.0;
            if (!parse_vec3_optional_z(iss, cur.center, defaultZ)) {
                throw std::runtime_error("Bad Center at line " + std::to_string(lineNo));
            }
            cur.hasCenter = true;
        } else if (key == "RotationCenter" || key == "RotateCenter" || key == "Pivot" || key == "RotationPivot") {
            double defaultZ = scene.dimension == 2 ? 0.5 * (scene.particleDistance > 0.0 ? scene.particleDistance : 1.0) : 0.0;
            if (!parse_vec3_optional_z(iss, cur.rotationPivot, defaultZ)) {
                throw std::runtime_error("Bad RotationCenter/Pivot at line " + std::to_string(lineNo));
            }
            cur.hasRotationPivot = true;
        } else if (key == "Size") {
            double defaultZ = scene.dimension == 2 ? (scene.particleDistance > 0.0 ? scene.particleDistance : 1.0) : 0.0;
            if (!parse_vec3_optional_z(iss, cur.size, defaultZ)) {
                throw std::runtime_error("Bad Size at line " + std::to_string(lineNo));
            }
            cur.hasSize = true;
        } else if (key == "Lower" || key == "Min") {
            Vec3 lo{};
            if (!parse_vec3_optional_z(iss, lo, 0.0)) {
                throw std::runtime_error("Bad Lower at line " + std::to_string(lineNo));
            }
            cur.xr.lo = lo.x; cur.xr.has = true;
            cur.yr.lo = lo.y; cur.yr.has = true;
            cur.zr.lo = lo.z; cur.zr.has = true;
        } else if (key == "Upper" || key == "Max") {
            Vec3 hi{};
            double defaultZ = scene.dimension == 2 ? (scene.particleDistance > 0.0 ? scene.particleDistance : 1.0) : 0.0;
            if (!parse_vec3_optional_z(iss, hi, defaultZ)) {
                throw std::runtime_error("Bad Upper at line " + std::to_string(lineNo));
            }
            cur.xr.hi = hi.x; cur.xr.has = true;
            cur.yr.hi = hi.y; cur.yr.has = true;
            cur.zr.hi = hi.z; cur.zr.has = true;
        } else if (key == "Radius") {
            iss >> cur.radius;
        } else if (key == "InnerRadius") {
            iss >> cur.innerRadius;
        } else if (key == "Ratio") {
            iss >> cur.ratio;
        } else if (key == "Height") {
            iss >> cur.height;
        } else if (key == "XRange") {
            iss >> cur.xr.lo >> cur.xr.hi;
            cur.xr.has = true;
        } else if (key == "YRange") {
            iss >> cur.yr.lo >> cur.yr.hi;
            cur.yr.has = true;
        } else if (key == "ZRange") {
            iss >> cur.zr.lo >> cur.zr.hi;
            cur.zr.has = true;
        } else if (key == "ZLayer") {
            iss >> cur.zLayerValue;
            cur.zSingleLayer = true;
        } else if (key == "P1") {
            if (!parse_vec2(iss, cur.p1)) {
                throw std::runtime_error("Bad P1 at line " + std::to_string(lineNo));
            }
            cur.hasTriangle = true;
        } else if (key == "P2") {
            if (!parse_vec2(iss, cur.p2)) {
                throw std::runtime_error("Bad P2 at line " + std::to_string(lineNo));
            }
            cur.hasTriangle = true;
        } else if (key == "P3") {
            if (!parse_vec2(iss, cur.p3)) {
                throw std::runtime_error("Bad P3 at line " + std::to_string(lineNo));
            }
            cur.hasTriangle = true;
        } else {
            throw std::runtime_error("Unknown shape/cuboid key at line " +
                                     std::to_string(lineNo) + ": " + key);
        }
    }

    if (inShape) {
        throw std::runtime_error("Unclosed Shape/Cuboid block at end of file");
    }

    if (scene.particleDistance <= 0.0) {
        throw std::runtime_error("ParticleDistance must be > 0");
    }

    if (scene.dimension == 2) {
        if (!scene.lowerSet) {
            scene.lowerDomain.z = 0.0;
            scene.lowerSet = true;
        }

        if (!scene.upperSet) {
            scene.upperDomain.z = scene.particleDistance;
            scene.upperSet = true;
        }
    }

    if (!scene.lowerSet || !scene.upperSet) {
        throw std::runtime_error("Domain must be set");
    }

    return scene;
}

static bool point_in_triangle_2d(const Vec2& p, const Vec2& a, const Vec2& b, const Vec2& c) {
    auto cross = [](const Vec2& u, const Vec2& v, const Vec2& w) {
        return (v.x - u.x) * (w.y - u.y) - (v.y - u.y) * (w.x - u.x);
    };

    double c1 = cross(a, b, p);
    double c2 = cross(b, c, p);
    double c3 = cross(c, a, p);

    bool hasNeg = (c1 < 0.0) || (c2 < 0.0) || (c3 < 0.0);
    bool hasPos = (c1 > 0.0) || (c2 > 0.0) || (c3 > 0.0);

    return !(hasNeg && hasPos);
}

static void add_particle_if_inside_domain(
    std::vector<Particle>& out,
    const Scene& scene,
    int type,
    int rigidType,
    const Vec3& p,
    const Vec3& vel,
    double enthalpy
) {
    if (p.x < scene.lowerDomain.x || p.x > scene.upperDomain.x) return;
    if (p.y < scene.lowerDomain.y || p.y > scene.upperDomain.y) return;
    if (p.z < scene.lowerDomain.z || p.z > scene.upperDomain.z) return;

    Particle q;
    q.type = type;
    q.rigidType = rigidType;
    q.pos = p;
    q.vel = vel;
    q.enthalpy = enthalpy;

    out.push_back(q);
}

static bool point_inside_rotated_box(const Particle& p, const Shape& s) {
    const Mat3 R = rotation_xyz_deg(s.rotateDeg);
    const Mat3 Rt = transpose(R);  // inverse rotation for a rotation matrix

    // Rotation is performed around the pivot.
    // Convert the world point back to the unrotated box coordinate.
    const Vec3 pivot = s.rotationPivot;
    const Vec3 unrotated = mul(Rt, p.pos - pivot) + pivot;

    const double eps = 1.0e-12;
    return unrotated.x >= s.xr.lo - eps && unrotated.x <= s.xr.hi + eps &&
           unrotated.y >= s.yr.lo - eps && unrotated.y <= s.yr.hi + eps &&
           unrotated.z >= s.zr.lo - eps && unrotated.z <= s.zr.hi + eps;
}

static void remove_particles_inside_box(
    std::vector<Particle>& out,
    const Shape& s
) {
    out.erase(
        std::remove_if(out.begin(), out.end(),
            [&](const Particle& p) {
                return point_inside_rotated_box(p, s);
            }),
        out.end()
    );
}

static std::vector<Particle> generate_particles(const Scene& scene) {
    std::vector<Particle> out;
    const double base_dx = scene.particleDistance;

    for (const Shape& s : scene.shapes) {
        const double lx = s.xr.lo;
        const double ux = s.xr.hi;
        const double ly = s.yr.lo;
        const double uy = s.yr.hi;
        const double lz = s.zr.lo;
        const double uz = s.zr.hi;

        const double wx = ux - lx;
        const double wy = uy - ly;
        const double wz = uz - lz;

        if (wx <= 0.0 || wy <= 0.0 || wz <= 0.0) {
            throw std::runtime_error("Shape has non-positive width: " + s.name);
        }

        int nx = (int)std::round(wx / base_dx);
        int ny = (int)std::round(wy / base_dx);
        int nz = (int)std::round(wz / base_dx);

        if (nx <= 0) nx = 1;
        if (ny <= 0) ny = 1;
        if (nz <= 0) nz = 1;

        const double target_sx = (s.hasSpacing && s.spacing.x > 0.0) ? s.spacing.x : base_dx;
        const double target_sy = (s.hasSpacing && s.spacing.y > 0.0) ? s.spacing.y : base_dx;
        const double target_sz = (s.hasSpacing && s.spacing.z > 0.0) ? s.spacing.z : base_dx;

        nx = (int)std::round(wx / target_sx);
        ny = (int)std::round(wy / target_sy);
        nz = (int)std::round(wz / target_sz);

        if (nx <= 0) nx = 1;
        if (ny <= 0) ny = 1;
        if (nz <= 0) nz = 1;

        const double sx = wx / nx;
        const double sy = wy / ny;
        const double sz = wz / nz;

        const Vec3 center = s.center;

        if (s.mode == ShapeMode::SUBTRACT) {
            if (s.kind == ShapeKind::BOX) {
                remove_particles_inside_box(out, s);
            }
            continue;
        }

        if (s.kind == ShapeKind::BOX) {
            const Mat3 R = rotation_xyz_deg(s.rotateDeg);
            const Vec3 pivot = s.rotationPivot;

            for (double x = lx + 0.5 * sx; x < ux - 0.49 * sx; x += sx) {
                for (double y = ly + 0.5 * sy; y < uy - 0.49 * sy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        Vec3 unrotated{x, y, z};
                        Vec3 world = mul(R, unrotated - pivot) + pivot;

                        add_particle_if_inside_domain(
                            out, scene, s.type, s.rigidType,
                            world, s.velocity, s.enthalpy
                        );
                    }
                }
            }
        }

        else if (s.kind == ShapeKind::SPHERE) {
            const double outerRadius = s.radius > 0.0 ? s.radius : 0.5 * wx;
            const double outer_r2 = outerRadius * outerRadius;
            const double inner_r2 = s.innerRadius * s.innerRadius;

            for (double x = lx + 0.5 * sx; x < ux - 0.49 * sx; x += sx) {
                for (double y = ly + 0.5 * sy; y < uy - 0.49 * sy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        double dx0 = x - center.x;
                        double dy0 = y - center.y;
                        double dz0 = z - center.z;
                        double r2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;

                        if (r2 > inner_r2 && r2 <= outer_r2) {
                            add_particle_if_inside_domain(
                                out, scene, s.type, s.rigidType,
                                {x, y, z}, s.velocity, s.enthalpy
                            );
                        }
                    }
                }
            }
        }

        else if (s.kind == ShapeKind::CYLINDER) {
            const double outerRadius = s.radius > 0.0 ? s.radius : 0.5 * wx;
            const double outer_r2 = outerRadius * outerRadius;

            for (double x = lx + 0.5 * sx; x < ux - 0.49 * sx; x += sx) {
                for (double y = ly + 0.5 * sy; y < uy - 0.49 * sy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        double dx0 = x - center.x;
                        double dy0 = y - center.y;
                        double r2 = dx0 * dx0 + dy0 * dy0;

                        if (r2 <= outer_r2) {
                            add_particle_if_inside_domain(
                                out, scene, s.type, s.rigidType,
                                {x, y, z}, s.velocity, s.enthalpy
                            );
                        }
                    }
                }
            }
        }

        else if (s.kind == ShapeKind::RING2D) {
            const double outerRadius = s.radius > 0.0 ? s.radius : 0.5 * wx;
            const double innerRadius = s.innerRadius > 0.0 ? s.innerRadius : 0.0;

            const double outer_r2 = outerRadius * outerRadius;
            const double inner_r2 = innerRadius * innerRadius;

            for (double x = lx + 0.5 * sx; x < ux - 0.49 * sx; x += sx) {
                for (double y = ly + 0.5 * sy; y < uy - 0.49 * sy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        double dx0 = x - center.x;
                        double dy0 = y - center.y;
                        double r2 = dx0 * dx0 + dy0 * dy0;

                        if (r2 > inner_r2 && r2 <= outer_r2) {
                            add_particle_if_inside_domain(
                                out, scene, s.type, s.rigidType,
                                {x, y, z}, s.velocity, s.enthalpy
                            );
                        }
                    }
                }
            }
        }

        else if (s.kind == ShapeKind::TRIANGLE) {
            for (double x = lx + 0.01 * sx; x < ux; x += sx) {
                for (double y = ly + 0.01 * sy; y < uy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        Vec2 p2{x, y};

                        if (point_in_triangle_2d(p2, s.p1, s.p2, s.p3)) {
                            add_particle_if_inside_domain(
                                out, scene, s.type, s.rigidType,
                                {x, y, z}, s.velocity, s.enthalpy
                            );
                        }
                    }
                }
            }
        }

        else if (s.kind == ShapeKind::TRIANGLE_PRISM) {
            Mat3 R = rotation_xyz_deg(s.rotateDeg);

            for (double x = lx + 0.01 * sx; x < ux; x += sx) {
                for (double y = ly + 0.01 * sy; y < uy; y += sy) {
                    for (double z = lz + 0.5 * sz; z < uz - 0.49 * sz; z += sz) {
                        Vec2 p2{x, y};

                        if (!point_in_triangle_2d(p2, s.p1, s.p2, s.p3)) {
                            continue;
                        }

                        Vec3 local{x - center.x, y - center.y, z - center.z};
                        Vec3 rotated = mul(R, local);
                        Vec3 world = rotated + center;

                        add_particle_if_inside_domain(
                            out, scene, s.type, s.rigidType,
                            world, s.velocity, s.enthalpy
                        );
                    }
                }
            }
        }
    }

    return out;
}

static void write_grid(
    const std::string& filename,
    const Scene& scene,
    const std::vector<Particle>& particles
) {
    std::ofstream ofs(filename.c_str());

    if (!ofs) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    ofs << std::setprecision(16);
    ofs << 0.0 << "\n";

    ofs << particles.size() << " "
        << scene.particleDistance << " "
        << scene.lowerDomain.x << " " << scene.upperDomain.x << " "
        << scene.lowerDomain.y << " " << scene.upperDomain.y << " "
        << scene.lowerDomain.z << " " << scene.upperDomain.z << "\n";

    for (const auto& p : particles) {
        ofs << p.type << " "
            << p.pos.x << " " << p.pos.y << " " << p.pos.z << " "
            << p.pos.x << " " << p.pos.y << " " << p.pos.z << " "
            << p.vel.x << " " << p.vel.y << " " << p.vel.z << "\n";
    }
}

static std::string stem_without_ext(const std::string& name) {
    size_t pos = name.find_last_of("/\\");
    std::string base = (pos == std::string::npos) ? name : name.substr(pos + 1);

    size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

static void write_svg_preview(
    const std::string& filename,
    const std::vector<Particle>& particles,
    char axisH,
    char axisV,
    double dx,
    const Scene& scene
) {
    double minH, maxH, minV, maxV;

    if (axisH == 'x') {
        minH = scene.lowerDomain.x;
        maxH = scene.upperDomain.x;
    } else if (axisH == 'y') {
        minH = scene.lowerDomain.y;
        maxH = scene.upperDomain.y;
    } else {
        minH = scene.lowerDomain.z;
        maxH = scene.upperDomain.z;
    }

    if (axisV == 'x') {
        minV = scene.lowerDomain.x;
        maxV = scene.upperDomain.x;
    } else if (axisV == 'y') {
        minV = scene.lowerDomain.y;
        maxV = scene.upperDomain.y;
    } else {
        minV = scene.lowerDomain.z;
        maxV = scene.upperDomain.z;
    }

    const int W = 1200;
    const int H = 800;
    const int pad = 50;

    auto get_coord = [](const Particle& p, char a) {
        if (a == 'x') return p.pos.x;
        if (a == 'y') return p.pos.y;
        return p.pos.z;
    };

    auto color_of = [](int type) {
        switch (type) {
            case 2: return std::string("#1f77b4");
            case 3: return std::string("#2ca02c");
            case 4: return std::string("#444444");
            case 5: return std::string("#d62728");
            default: return std::string("#9467bd");
        }
    };

    std::ofstream ofs(filename.c_str());

    ofs << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W
        << "' height='" << H << "' viewBox='0 0 " << W << " " << H << "'>\n";

    ofs << "<rect width='100%' height='100%' fill='white'/>\n";
    ofs << "<rect x='" << pad << "' y='" << pad
        << "' width='" << (W - 2 * pad)
        << "' height='" << (H - 2 * pad)
        << "' fill='none' stroke='black' stroke-width='1'/>\n";

    for (const auto& p : particles) {
        double h = get_coord(p, axisH);
        double v = get_coord(p, axisV);

        double sx = pad + (h - minH) / (maxH - minH) * (W - 2 * pad);
        double sy = H - pad - (v - minV) / (maxV - minV) * (H - 2 * pad);

        double rr = std::max(
            1.0,
            0.35 * dx / std::max(maxH - minH, maxV - minV) *
            std::min(W - 2 * pad, H - 2 * pad)
        );

        ofs << "<circle cx='" << sx << "' cy='" << sy << "' r='" << rr
            << "' fill='" << color_of(p.type) << "' fill-opacity='0.85'/>\n";
    }

    ofs << "<text x='" << pad << "' y='25' font-size='20'>Preview "
        << axisH << axisV << " plane</text>\n";

    ofs << "<text x='" << W / 2 << "' y='" << H - 10
        << "' font-size='16'>" << axisH << "</text>\n";

    ofs << "<text x='10' y='" << H / 2
        << "' font-size='16' transform='rotate(-90 10,"
        << H / 2 << ")'>" << axisV << "</text>\n";

    ofs << "</svg>\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: " << argv[0] << " DEM   or   "
                      << argv[0] << " DEM.boid\n";
            return 1;
        }

        std::string inputArg = argv[1];
        std::string inputFile;
        std::string outputStem;

        if (inputArg.size() >= 5 &&
            inputArg.substr(inputArg.size() - 5) == ".boid") {
            inputFile = inputArg;
            outputStem = stem_without_ext(inputArg);
        } else {
            inputFile = inputArg + ".boid";
            outputStem = inputArg;
        }

        Scene scene = read_scene(inputFile);
        std::vector<Particle> particles = generate_particles(scene);

        std::string gridFile = outputStem + ".grid";
        std::string xyFile = outputStem + "_preview_xy.svg";
        std::string xzFile = outputStem + "_preview_xz.svg";

        write_grid(gridFile, scene, particles);
        write_svg_preview(xyFile, particles, 'x', 'y', scene.particleDistance, scene);
        write_svg_preview(xzFile, particles, 'x', 'z', scene.particleDistance, scene);

        std::cout << particles.size() << " particles were generated\n";
        std::cout << "Input : " << inputFile << "\n";
        std::cout << "Output: " << gridFile << "\n";
        std::cout << "Preview XY: " << xyFile << "\n";
        std::cout << "Preview XZ: " << xzFile << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}