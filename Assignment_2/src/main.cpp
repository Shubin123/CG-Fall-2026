// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

void raytrace_sphere()
{
    std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

    const std::string filename("sphere_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the
    // unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    const double sphere_radius = 0.9;
    const Vector3d sphere_center(0, 0, 0);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Generic ray-sphere intersection
            // Ray: p(t) = ray_origin + t * ray_direction
            // Sphere: ||p - sphere_center||^2 = sphere_radius^2
            const Vector3d oc = ray_origin - sphere_center;
            const double a = ray_direction.dot(ray_direction);
            const double b = 2.0 * ray_direction.dot(oc);
            const double c = oc.dot(oc) - sphere_radius * sphere_radius;
            const double discriminant = b * b - 4.0 * a * c;

            if (discriminant >= 0)
            {
                // Nearest intersection (smallest positive t)
                const double t = (-b - sqrt(discriminant)) / (2.0 * a);

                // The ray hit the sphere, compute the exact intersection point
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_parallelogram()
{
    std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

    const std::string filename("plane_orthographic.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Prepare the ray
            const Vector3d ray_origin = pixel_center;
            const Vector3d ray_direction = camera_view_direction;

            // Ray-parallelogram intersection
            // Solve: ray_origin + t * ray_direction = pgram_origin + u * pgram_u + v * pgram_v
            // Rearrange: [pgram_u | pgram_v | -ray_direction] * [u; v; t] = ray_origin - pgram_origin
            Matrix3d M;
            M.col(0) = pgram_u;
            M.col(1) = pgram_v;
            M.col(2) = -ray_direction;

            const Vector3d rhs = ray_origin - pgram_origin;
            const Vector3d uvt = M.colPivHouseholderQr().solve(rhs);
            const double u_param = uvt(0);
            const double v_param = uvt(1);
            const double t = uvt(2);

            // Check that the intersection is inside the parallelogram and in front of the ray
            if (u_param >= 0 && u_param <= 1 && v_param >= 0 && v_param <= 1 && t > 0)
            {
                // The ray hit the parallelogram, compute the exact intersection point
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute normal at the intersection point (constant across the parallelogram)
                Vector3d ray_normal = pgram_u.cross(pgram_v).normalized();

                // Ensure the normal faces toward the camera
                if (ray_normal.dot(-ray_direction) < 0)
                    ray_normal = -ray_normal;

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_perspective()
{
    std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

    const std::string filename("plane_perspective.png");
    MatrixXd C = MatrixXd::Zero(800, 800); // Store the color
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / C.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / C.rows(), 0);

    // Parameters of the parallelogram (position of the lower-left corner + two sides)
    const Vector3d pgram_origin(-0.5, -0.5, 0);
    const Vector3d pgram_u(1, 0.4, 0);
    const Vector3d pgram_v(0, 0.7, -10);

    // Single light source
    const Vector3d light_position(-1, 1, 1);

    for (unsigned i = 0; i < C.cols(); ++i)
    {
        for (unsigned j = 0; j < C.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Perspective: ray originates from camera, direction goes through pixel center
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();

            // Ray-parallelogram intersection
            Matrix3d M;
            M.col(0) = pgram_u;
            M.col(1) = pgram_v;
            M.col(2) = -ray_direction;

            const Vector3d rhs = ray_origin - pgram_origin;
            const Vector3d uvt = M.colPivHouseholderQr().solve(rhs);
            const double u_param = uvt(0);
            const double v_param = uvt(1);
            const double t = uvt(2);

            if (u_param >= 0 && u_param <= 1 && v_param >= 0 && v_param <= 1 && t > 0)
            {
                // The ray hit the parallelogram, compute the exact intersection point
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute normal at the intersection point
                Vector3d ray_normal = pgram_u.cross(pgram_v).normalized();

                // Ensure the normal faces toward the camera
                if (ray_normal.dot(-ray_direction) < 0)
                    ray_normal = -ray_normal;

                // Simple diffuse model
                C(i, j) = (light_position - ray_intersection).normalized().transpose() * ray_normal;

                // Clamp to zero
                C(i, j) = std::max(C(i, j), 0.);

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(C, C, C, A, filename);
}

void raytrace_shading()
{
    std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

    const std::string filename("shading.png");
    MatrixXd R = MatrixXd::Zero(800, 800); // Store red channel
    MatrixXd G = MatrixXd::Zero(800, 800); // Store green channel
    MatrixXd B = MatrixXd::Zero(800, 800); // Store blue channel
    MatrixXd A = MatrixXd::Zero(800, 800); // Store the alpha mask

    const Vector3d camera_origin(0, 0, 3);
    const Vector3d camera_view_direction(0, 0, -1);

    // The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
    const Vector3d image_origin(-1, 1, 1);
    const Vector3d x_displacement(2.0 / A.cols(), 0, 0);
    const Vector3d y_displacement(0, -2.0 / A.rows(), 0);

    //Sphere setup
    const Vector3d sphere_center(0, 0, 0);
    const double sphere_radius = 0.9;

    //material params
    const Vector3d diffuse_color(1, 0, 1);
    const double specular_exponent = 100;
    const Vector3d specular_color(0., 0, 1);

    // Single light source
    const Vector3d light_position(-1, 1, 1);
    const Vector3d light_intensity(1, 1, 1);
    double ambient = 0.1;

    for (unsigned i = 0; i < A.cols(); ++i)
    {
        for (unsigned j = 0; j < A.rows(); ++j)
        {
            const Vector3d pixel_center = image_origin + double(i) * x_displacement + double(j) * y_displacement;

            // Perspective ray
            const Vector3d ray_origin = camera_origin;
            const Vector3d ray_direction = (pixel_center - camera_origin).normalized();

            // Generic ray-sphere intersection
            const Vector3d oc = ray_origin - sphere_center;
            const double a = ray_direction.dot(ray_direction);
            const double b_coeff = 2.0 * ray_direction.dot(oc);
            const double c = oc.dot(oc) - sphere_radius * sphere_radius;
            const double discriminant = b_coeff * b_coeff - 4.0 * a * c;

            if (discriminant >= 0)
            {
                const double t = (-b_coeff - sqrt(discriminant)) / (2.0 * a);

                // Compute exact intersection point
                Vector3d ray_intersection = ray_origin + t * ray_direction;

                // Compute normal at the intersection point
                Vector3d ray_normal = (ray_intersection - sphere_center).normalized();

                // Blinn-Phong shading
                const Vector3d l = (light_position - ray_intersection).normalized(); // light direction
                const Vector3d v = (camera_origin - ray_intersection).normalized();  // view direction
                const Vector3d h = (l + v).normalized();                             // half-vector

                const double diffuse = std::max(0.0, ray_normal.dot(l));
                const double specular = std::pow(std::max(0.0, ray_normal.dot(h)), specular_exponent);

                // Per-channel color: ambient + diffuse + specular
                R(i, j) = std::max(0.0, ambient * diffuse_color(0) + diffuse * diffuse_color(0) * light_intensity(0) + specular * specular_color(0) * light_intensity(0));
                G(i, j) = std::max(0.0, ambient * diffuse_color(1) + diffuse * diffuse_color(1) * light_intensity(1) + specular * specular_color(1) * light_intensity(1));
                B(i, j) = std::max(0.0, ambient * diffuse_color(2) + diffuse * diffuse_color(2) * light_intensity(2) + specular * specular_color(2) * light_intensity(2));

                // Disable the alpha mask for this pixel
                A(i, j) = 1;
            }
        }
    }

    // Save to png
    write_matrix_to_png(R, G, B, A, filename);
}

int main()
{
    raytrace_sphere();
    raytrace_parallelogram();
    raytrace_perspective();
    raytrace_shading();

    return 0;
}