import numpy as np

def least_squares_fit(data, x, y, z):
    # Calculate the spherical harmonic functions at the points on the sphere
    harmonics = spherical_harmonics(x, y, z)

    # Convert the input arrays to at least two dimensions
    harmonics = np.atleast_2d(harmonics)
    data = np.atleast_2d(data)

    # Solve the linear system of equations to calculate the spherical harmonic
    # coefficients that minimize the difference between the data and its approximation
    coefficients, _, _, _ = np.linalg.lstsq(harmonics, data, rcond=None)

    return coefficients

def spherical_harmonics(x, y, z):
    # Calculate the spherical coordinates of the points on the sphere
    theta = np.arctan2(y, x)
    phi = np.arccos(z)

    # Calculate the spherical harmonic functions at the given points on the sphere
    Y00 = 0.5 * np.sqrt(1/np.pi)
    Y10 = 0.5 * np.sqrt(3/np.pi) * np.cos(theta)
    Y11 = 0.5 * np.sqrt(3/np.pi) * np.sin(theta) * np.cos(phi)
    Y20 = 0.5 * np.sqrt(5/np.pi) * 0.5 * (3*np.cos(theta)**2 - 1)
    Y21 = 0.5 * np.sqrt(15/np.pi) * np.sin(theta) * np.cos(theta) * np.cos(phi)
    Y22 = 0.5 * np.sqrt(15/np.pi) * np.sin(theta) * np.sin(phi)

    # Return the spherical harmonic functions as a matrix
    return np.array([Y00, Y10, Y11, Y20, Y21, Y22])

def spherical_harmonic_approximation(coefficients, x, y, z):
    # Calculate the spherical harmonic functions at the points on the sphere
    harmonics = spherical_harmonics(x, y, z)

    # Use the coefficients to combine the spherical harmonic functions
    # into an approximation of the original data
    approximation = np.dot(harmonics, coefficients)

    return approximation

def sample_data(x, y, z):
    # Define the function to be sampled
    def f(x, y, z):
        return x**2 + y**2 + z**2

    # Sample the function at the points on the sphere
    data = f(x, y, z)

    return data

# Define the number of points on the sphere where the data will be sampled
n = 100

# Generate a set of points on the surface of the sphere using
# the spherical coordinates (theta, phi)
theta = np.linspace(0, 2*np.pi, n)
phi = np.linspace(0, np.pi, n)

# Calculate the Cartesian coordinates of the points on the sphere
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Sample the data at the points on the sphere
data = sample_data(x, y, z)

# Calculate the spherical harmonic coefficients using least squares fitting
coefficients = least_squares_fit(data, x, y, z)

# Use the coefficients to reconstruct an approximation of the original data
reconstructed_data = spherical_harmonic_approximation(coefficients, x, y, z)

print(data - reconstructed_data)
