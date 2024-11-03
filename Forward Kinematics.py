import numpy as np
import math
from scipy.linalg import expm

pi=np.pi

def skew(vector):
    """Generate a skew-symmetric matrix from a 3D vector."""
    x, y, z = vector
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def adjoint_twist(angular, linear):
    """Form the twist matrix in SE(3) from angular and linear components."""
    omega_hat = skew(angular)  # 3x3 skew-symmetric matrix for rotation
    v = np.array(linear).reshape(3, 1)  # Linear part as 3x1
    twist = np.zeros((4, 4))
    twist[:3, :3] = omega_hat
    twist[:3, 3] = v.squeeze()
    return twist

def screw_exponential(angular, linear, theta):
    """
    Compute the matrix exponential for a screw transformation in SE(3).
    
    Parameters:
    - angular: 3D vector representing angular velocity
    - linear: 3D vector representing linear velocity
    - theta: Scalar rotation angle in radians
    
    Returns:
    - The transformation matrix (4x4) in SE(3)
    """
    twist_matrix = adjoint_twist(angular, linear)
    transformation = expm(twist_matrix * theta)
    return transformation

M = np.array([[1,0,0,3.732],[0,1,0,0],[0,0,1,2.732],[0,0,0,1]])
Theta = np.array([-pi/2, pi/2, pi/3, -pi/4, 1, pi/6])

s = np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],[0,2.732,3.732,2,0,0],[2.732,0,0,0,0,0],[0,-2.732,-1,0,1,0]]) # defined in body frame

# Body frame calculation
M_body = M.copy()
for i in range(len(s.T)):
    s_i = s[:, i]
    angular_velocity = s_i[:3]
    linear_velocity = s_i[3:]
    theta = Theta[i]
    M_body = M_body @ screw_exponential(angular_velocity, linear_velocity, theta)
print("End effector:\n", M_body.round(2))


s = np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],[0,0,1,-0.732,0,0],[-1,0,0,0,0,-3.732],[0,1,2.732,3.732,1,0]]) # defined in space frame

# defined in space frame
M_space = M.copy()
for i in range(len(s.T) - 1, -1, -1):
    s_i = s[:, i]
    angular_velocity = s_i[:3]
    linear_velocity = s_i[3:]
    theta = Theta[i]
    M_space = screw_exponential(angular_velocity, linear_velocity, theta) @ M_space
print("End effector:\n", M_space.round(2))