////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
// Shortcut to avoid  everywhere, DO NOT USE IN .h
using namespace Eigen;
////////////////////////////////////////////////////////////////////////////////
#define DATA_DIR "../data"
const std::string root_path = DATA_DIR;

// Computes the determinant of the matrix whose columns are the vector u and v
double inline det(const Vector2d &u, const Vector2d &v)
{
    double a = u[0] * v[1];
    double b = u[1] * v[0];
    return a - b;
}

// Return true iff [a,b] intersects [c,d]
bool intersect_segment(const Vector2d &a,
                       const Vector2d &b,
                       const Vector2d &c,
                       const Vector2d &d)
{
    // Direction vectors
    Vector2d ab = b - a;
    Vector2d ac = c - a;
    Vector2d ad = d - a;

    Vector2d cd = d - c;
    Vector2d ca = a - c;
    Vector2d cb = b - c;

    double d1 = det(ab, ac);
    double d2 = det(ab, ad);
    double d3 = det(cd, ca);
    double d4 = det(cd, cb);

    // General case:
    // c and d are on opposite sides of ab
    // a and b are on opposite sides of cd
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
    {
        return true;
    }

    // Handle collinear cases
    auto on_segment = [](const Vector2d &p,
                         const Vector2d &q,
                         const Vector2d &r)
    {
        // q lies on segment [p,r]
        return q[0] >= std::min(p[0], r[0]) &&
               q[0] <= std::max(p[0], r[0]) &&
               q[1] >= std::min(p[1], r[1]) &&
               q[1] <= std::max(p[1], r[1]);
    };

    const double EPS = 1e-12;

    if (std::abs(d1) < EPS && on_segment(a, c, b))
        return true;

    if (std::abs(d2) < EPS && on_segment(a, d, b))
        return true;

    if (std::abs(d3) < EPS && on_segment(c, a, d))
        return true;

    if (std::abs(d4) < EPS && on_segment(c, b, d))
        return true;

    return false;
}

////////////////////////////////////////////////////////////////////////////////

bool is_inside(const std::vector<Vector2d> &poly,
               const Vector2d &query)
{
    if (poly.size() < 3)
        return false;

    // 1. Compute bounding box
    double min_x = poly[0][0];
    double max_x = poly[0][0];
    double min_y = poly[0][1];
    double max_y = poly[0][1];

    for (const auto &p : poly)
    {
        min_x = std::min(min_x, p[0]);
        max_x = std::max(max_x, p[0]);
        min_y = std::min(min_y, p[1]);
        max_y = std::max(max_y, p[1]);
    }

    // Pick a point definitely outside the bounding box.
    Vector2d outside(max_x + 1.0, max_y + 1.0);

    // 2. Cast a ray from query to outside.
    int intersections = 0;

    for (size_t i = 0; i < poly.size(); ++i)
    {
        Vector2d a = poly[i];
        Vector2d b = poly[(i + 1) % poly.size()];

        // If query is exactly on an edge, consider it inside.
        Vector2d edge = b - a;
        Vector2d to_query = query - a;

        if (std::abs(det(edge, to_query)) < 1e-12)
        {
            if (query[0] >= std::min(a[0], b[0]) &&
                query[0] <= std::max(a[0], b[0]) &&
                query[1] >= std::min(a[1], b[1]) &&
                query[1] <= std::max(a[1], b[1]))
            {
                return true;
            }
        }

        if (intersect_segment(query, outside, a, b))
        {
            ++intersections;
        }
    }

    // Odd number of crossings => inside.
    return (intersections % 2) == 1;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<Vector2d> load_xyz(const std::string &filename)
{
    std::vector<Vector2d> points;
    std::ifstream in(filename);

    // Check if file is open
    if (!in.is_open())
    {
        std::cerr << "Error: could not open " << filename << std::endl;
        return points;
    }

    // Read x and y from each line.
    // If the file has x y z, z is ignored.
    double x, y;

    while (in >> x >> y)
    {
        points.push_back(Vector2d(x, y));

        // If xyz files contain a third coordinate, ignore it.
        // This assumes each point is represented by x y z.
        std::string remainder;
        std::getline(in, remainder);
    }

    return points;
}

void save_xyz(const std::string &filename,
              const std::vector<Vector2d> &points)
{
    std::ofstream out(filename);

    if (!out.is_open())
    {
        std::cerr << "Error: could not open " << filename << std::endl;
        return;
    }

    for (const auto &p : points)
    {
        out << p[0] << " " << p[1] << std::endl;
    }
}

std::vector<Vector2d> load_obj(const std::string &filename)
{
    std::ifstream in(filename);
    std::vector<Vector2d> points;
    std::vector<Vector2d> poly;

    if (!in.is_open())
    {
        std::cerr << "Error: could not open " << filename << std::endl;
        return poly;
    }

    char key;

    while (in >> key)
    {
        if (key == 'v')
        {
            double x, y, z;
            in >> x >> y >> z;

            points.push_back(Vector2d(x, y));
        }
        else if (key == 'f')
        {
            std::string line;
            std::getline(in, line);

            std::istringstream ss(line);

            int id;

            while (ss >> id)
            {
                poly.push_back(points[id - 1]);
            }
        }
    }

    return poly;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    const std::string points_path = root_path + "/points.xyz";
    const std::string poly_path = root_path + "/polygon.obj";

    std::vector<Vector2d> points = load_xyz(points_path);

    for (auto point : points)
    {
        std::cout << point << std::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Point in polygon

    std::vector<Vector2d> poly = load_obj(poly_path);

    std::vector<Vector2d> result;

    for (size_t i = 0; i < points.size(); ++i)
    {
        if (is_inside(poly, points[i]))
        {
            result.push_back(points[i]);
        }
    }

    save_xyz("output.xyz", result);

    ////////////////////////////////////////////////////////////////////////////////
    // Eigen matrix testing;
     
    // Matrix3d A;

    // A << 1, 2, 3,
    //      4, 5, 6,
    //      7, 8, 9;

    // auto B = A.colPivHouseholderQr();

    // std::cout << A << std::endl;

    return 0;
}