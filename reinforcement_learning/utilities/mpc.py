import numpy as np 
import pydrake.symbolic as sym #pydrake is not on Windows, test in Linux/WSL or macOS
import cvxpy as cp
import torch
import casadi as cas


class MPC:
    def __init__(self, Hp, Q, Qc, Kj, R, P, Ru, nx, nu, lr, wheelbase, max_steering_angle, max_speed, car_w, car_l, dt):
        self.Hp = Hp #prediction horizon
        self.Q = Q #state cost
        self.Qc = Qc # AA collision cost
        self.Kj = Kj #collision avoidance gain
        self.R = R #input cost
        self.P = P #input rate of change cost
        self.Ru = Ru #difference between input and MARL input cost
        self.nx = nx #state dimension
        self.nu = nu #input dimension
        self.lr = lr #distance from the center of gravity to the rear axle
        self.wheelbase = wheelbase #distance between the front and rear axles
        self.max_steering_angle = max_steering_angle #max steering angle
        self.max_speed = max_speed #max speed
        self.dt = dt #time step duration
        self.car_w = car_w #car width
        self.car_l = car_l #car length
        self.total_converged_solutions = 0
        self.total_solve_attempts = 0
        self.solve_attempt_time = 0
    
    
    def get_linear_matrices(self,x,u):
        #Adapted from: https://github.com/mschoder/vehicle_mpc/blob/main/dynamics.py 
        #Based on the kinematics bicycle model described in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9942280 (eq. 12)
        # TODO: make it run every iteration
        # [x, y, heading, velocity, steering_angle] = x (we use the streering angle u instead of the one in x to calculate the new heading)
        # [v_command, steering_angle_command] = u
        # we don't really care about the steering angle of the front wheel, we only care about the yaw which is the orientation/heading of the car
        """
        This function should return the linearized matrices A and B

        Args:
            x (np.array): state vector
            u (np.array): input vector

        Returns:
            A (np.array): linearized state matrix
            B (np.array): linearized input matrix
        """

        #Create symbolic variables
        x_sym = np.array([sym.Variable("x_{}".format(i))
                              for i in range(self.nx)])
        u_sym = np.array([sym.Variable("u_{}".format(i))
                              for i in range(self.nu)])

        #v = sym.sqrt(sym.add(x_sym[3]**2, x_sym[4]**2))
        heading = x_sym[2]
        steering_angle_command = u_sym[1]
        v_command = u_sym[0]

        #Compute the slip angle beta 
        beta = sym.arctan((self.lr/self.wheelbase)*sym.tan(steering_angle_command))

        #Compute the model dynamics array
        f = np.array([v_command*sym.cos(heading+beta), #x_dot
                      v_command*sym.sin(heading+beta), #y_dot
                      v_command*(1/self.wheelbase)*sym.cos(beta)*sym.tan(steering_angle_command), #yaw_dot
                      v_command, 
                      steering_angle_command]) 
        
        
        
        #Create symbolic function
        f_x = sym.Jacobian(f, x_sym)
        f_u = sym.Jacobian(f, u_sym)

        #Create mapping of the symbolic variables to the env variables
        mapping = {x_sym[i]: x[i] for i in range(self.nx)}
        mapping.update({u_sym[i]: u[i] for i in range(self.nu)})

        A = sym.Evaluate(f_x, mapping)
        B = sym.Evaluate(f_u, mapping)

        return A, B

    
    def solve(self, x0, marl_u_seq, target_states, slack_vars, other_vehicles_states, prev_cmd=None): 
        """
        This function should return the optimal control input for the current state,
        for the current vehicle, given the current target states and other vehicles' 
        predicted future states

        Args:
            x0: current state
            marl_u_seq: sequence of control inputs from MARL over Hp timesteps
            target_states: target states for the prediction horizon
            other_vehicles_states: predicted future states of other vehicles
            slack_vars (list): slack variables for collision avoidance
            prev_cmd: previous control input for the current vehicle

        Returns:
            other_vehicles_states: predicted future states of current and other vehicles
            optimal control input for the current vehicle OR the MARL control input if the MPC does not converge
        """

        #print("[DEBUG] Solving MPC")

        cost = 0
        constraints = []
        opti = cas.Opti()

        #Variables
        # x = cp.Variable((self.nx, self.Hp+1),name='x')
        # u = cp.Variable((self.nu, self.Hp),name='u')
        x = opti.variable(self.nx, self.Hp+1) 
        u = opti.variable(self.nu, self.Hp)

        x0_const = x0.detach().numpy()


        #Linearized matrices
        # for every timestep first get the rate of change of the actions from marl and pass it to the linearizor func
        # and call it at every iteration
        # A, B = self.get_linear_matrices(x0, marl_u_seq[0])
        A, B = self.get_linear_matrices(x0, prev_cmd)

        #Cost function and constraints
        for k in range(self.Hp):
            marl_u_seq_for_cost = np.array(marl_u_seq[k].detach().numpy(), dtype=np.float32)
            # target_states_for_cost = np.array(target_states[k].detach().numpy(), dtype=np.float32)

            #State cost
            # cost += (x[:,k] - target_states_for_cost).T @ self.Q @ (x[:,k] - target_states_for_cost)

            # #Control effort/Input cost
            # cost += (u[:,k]).T @ self.R @ (u[:,k])

            # difference between input and MARL input cost
            cost += (u[:,k] - marl_u_seq_for_cost).T @ self.Ru @ (u[:,k] - marl_u_seq_for_cost)
            
            #State cost
            # cost += cp.quad_form(x[:,k] - target_states[k], self.Q)
            
            # #Control effort/Input cost
            # cost += cp.quad_form(u[:,k], self.R)

            # #difference between input and MARL input cost
            # cost += cp.quad_form(u[:,k] - marl_u_seq[k], self.Ru)

            # if (k < self.Hp-1):
            #     #Control rate of change cost
            #     cost += (u[:,k+1] - u[:,k]).T @ self.P @ (u[:,k+1] - u[:,k])

                # #Control rate of change cost
                # cost += cp.quad_form(u[:,k+1] - u[:,k], self.P)

            # #Dynamics constraints
            # constraints += [x[:,k + 1] == A @ x[:,k] + B @ u[:,k]]
            # constraints += [x[:,k] >= xmin, x[:,k] <= xmax]

            # Dynamics constraints 
            constraints.append(x[:,k + 1] == A @ x[:,k] + B @ u[:,k])

            #Control Input constraints
            constraints.append(u[0,k] >= 0.0001)
            constraints.append(u[0,k] <= self.max_speed)
            constraints.append(u[1,k] >= -self.max_steering_angle)
            constraints.append(u[1,k] <= self.max_steering_angle)

            # #Input constraints
            # constraints += [u[0,k] >= 0, u[0,k] <= self.max_speed] 
            # constraints += [u[1,k] >= -self.max_steering_angle, u[1,k] <= self.max_steering_angle]
            

            
            # if k > 0:
            #     #Input rate of change constraints
            #     constraints += [cp.abs(u[0,k] - u[0,k-1]) / self.dt <= self.max_speed]
            #     constraints += [cp.abs(u[1,k] - u[1,k-1]) / self.dt <= self.max_steering_angle]
                
        k = 0

        # #Initial state constraints 
        # constraints += [x[:,0] == x0]

        # #Initial state constraints
        constraints.append(x[:,0] - x0_const == 0)
    


        #NOTE: if this is not the vehicle with the highest priority,
        #then it has the predicted states of all of the other vehicles preceding it 
        #and it should try to avoid them at each state
        if other_vehicles_states is not None:
            for i in range(len(other_vehicles_states)):
                for k in range(self.Hp):
                    # print("[DEBUG] Other vehicle states: ", other_vehicles_states[i][k][:2])
                    other_vehicles_state_tensor = torch.stack([other_vehicles_states[i][k][0], other_vehicles_states[i][k][1], other_vehicles_states[i][k][2]])
                    other_vehicles_state_tensor_for_calc = np.array(other_vehicles_state_tensor.detach().numpy(), dtype=np.float32)
                    # print("[DEBUG] Other vehicle state tensor: ", other_vehicles_state_tensor)
                    # distance = cp.norm(x[:2,k] - other_vehicles_state_tensor, 2)
                    distance = cas.norm_2(x[:2,k] - other_vehicles_state_tensor_for_calc[:2])



                    # Add collision avoidance cost
                    # source: https://www.researchgate.net/publication/314237612_Nonlinear_Model_Predictive_Control_for_Multi-Micro_Aerial_Vehicle_Robust_Collision_Avoidance
                    # cost += self.Qc + cp.exp(distance-(1-slack_vars[i][k]))
                    cost += self.Qc[k] / (1 + cas.exp(self.Kj*(distance - (1 - slack_vars[i][k]))))


        # prob = cp.Problem(cp.Minimize(cost), constraints)
        # prob.solve(method='dccp',ccp_times=1,max_iter=100,tau=0.005, mu=1.2, tau_max=1e8)
        # prob.solve()
        # print("     [DEBUG] Status:", prob.status)
        self.total_solve_attempts += 1

        # Define the NLP and solver
        opti.minimize(cost)
        opti.subject_to(constraints)
        p_opts = {"expand":True}
        s_opts = {"max_iter": 100}
        opts = {"qpsol": "qrqp"}


        # opti.solver('sqpmethod', opts)
        opti.solver('ipopt',p_opts,s_opts)
        # opti.solver('qrqp')
        # opti.solver('qpoases')

        try: 
            sol = opti.solve()
        except:
            print("[DEBUG] Status: Not Converged")
            #predict future states for next Hp timesteps
            next_states = [[]]
            # print("[DEBUG] value: ", u.value)
            for k in range(self.Hp):
                # print("[DEBUG] Predicting future state: ", u.value[:,k])
                next_states.append(self.predict_future_state(x0, marl_u_seq[k]))
                x0 = next_states[-1]
            next_states = next_states[1:]
            # print("[DEBUG] Next states: ", next_states)
            if other_vehicles_states is not None:
                other_vehicles_states.append(next_states)
            else:
                other_vehicles_states = [next_states]

            return marl_u_seq[0], other_vehicles_states
            # return prev_cmd, other_vehicles_states 


        #if prob.status == "Converged" or prob.status == "optimal":
        if opti.stats()['return_status'] == "Solve_Succeeded":
            print("[DEBUG] Status: Converged")
            self.total_converged_solutions += 1
            self.solve_attempt_time += opti.stats()['t_wall_total']
            #predict future states for next Hp timesteps
            next_states = [[]]
            # print("[DEBUG] value: ", u.value)
            for k in range(self.Hp):
                # print("[DEBUG] Predicting future state: ", u.value[:,k])
                # print("[DEBUG] Predicting future state: ", sol.value(u)[:,k])
                #next_states.append(self.predict_future_state(x0, u.value[:,k]))
                next_states.append(self.predict_future_state(x0, sol.value(u)[:,k]))
                x0 = next_states[-1]
            next_states = next_states[1:]
            # print("[DEBUG] Next states: ", next_states)
            if other_vehicles_states is not None:
                other_vehicles_states.append(next_states)
            else:
                other_vehicles_states = [next_states]
            # print("[DEBUG] Next states for all vehicles: ", other_vehicles_states)

            #Return the first control input
            # return torch.tensor([u.value[0,0],u.value[1,0]],dtype=torch.float32), other_vehicles_states
            return torch.tensor([sol.value(u)[0,0],sol.value(u)[1,0]],dtype=torch.float32), other_vehicles_states
        else:
            self.solve_attempt_time += opti.stats()['t_wall_total']
            # predict future states for next Hp timesteps
            next_states = [[]]
            # print("[DEBUG] value: ", u.value)
            for k in range(self.Hp):
                # print("[DEBUG] Predicting future state: ", u.value[:,k])
                next_states.append(self.predict_future_state(x0, marl_u_seq[k]))
                x0 = next_states[-1]
            next_states = next_states[1:]
            # print("[DEBUG] Next states: ", next_states)
            if other_vehicles_states is not None:
                other_vehicles_states.append(next_states)
            else:
                other_vehicles_states = [next_states]

            return marl_u_seq[0], other_vehicles_states
            # return prev_cmd, other_vehicles_states
        

    def predict_future_state(self, current_state, control_input):
        #Based on the kinematics bicycle model listed in Equation 12 of: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9942280
        #And implemented in the KinematicBicycle class in utilities/kinematic_bicycle.py
        """
        This function should return the predicted future state of the vehicle
        given the current state and control input

        Args:
            current_state: current state of the vehicle
            control_input: control input

        Returns:
            list: predicted future state
        """

        #TODO: handle the tensor user warnings

        #Extract the current state and control inputs
        x, y, heading, velocity, steering_angle = current_state
        v_command, steering_angle_command = control_input

        #Compute the slip angle beta 
        beta = torch.atan2((self.lr/self.wheelbase)*torch.tan(torch.tensor(steering_angle_command)),torch.tensor([1.0]))

        #Compute the model dynamics
        dx = v_command*torch.cos(heading+beta) # we use the old heading
        dy = v_command*torch.sin(heading+beta)
        dheading = v_command*(1/self.wheelbase)*torch.cos(beta)*torch.tan(torch.tensor(steering_angle_command)) 
        

        future_state = [x+dx*self.dt, y+dy*self.dt, heading+dheading*self.dt, v_command, steering_angle_command]

        return future_state