import numpy as np
import matplotlib.pyplot as plt
import math as math

class Model:
    initial_states = {"x":0,"v":0,"omega":0,"omega_dot":0}
    action_force = {"left":-10,"right":10}
    reward = 1
    gamma = 1
    
class ModelDynamics:
    gravity = 9.8
    cart_mass = 1
    pole_mass = 0.1
    total_mass = 1.1
    pole_length = 0.5
    action_interval = 0.02
    
class Model:
    initial_states = {"x":0,"v":0,"omega":0,"omega_dot":0}
    action_force = {"left":-10,"right":10}
    reward = 1
    gamma = 1
    
class Hyperparameters:
    num_perturbations = 20
    N = 1
    sigma = 0.5
    alpha = 0.001
    M = 5
    
class CartPole:
    def evolution_strategies():
        j_final = []
        policy_dim = (4*Hyperparameters.M)+1
        initial_policy = np.random.rand(policy_dim,1)
        current_policy = initial_policy
        for i in range(100):
            total_j_epsilon = 0
            # print("Policy values",current_policy)
            for j in range(Hyperparameters.num_perturbations):
                perturbation = np.random.multivariate_normal(mean=np.zeros(policy_dim),cov=np.identity(policy_dim)).reshape(policy_dim,1)
                perturbed_policy = current_policy + (Hyperparameters.sigma * perturbation)
                reward_returned = CartPole.estimate_j(perturbed_policy)
                #print('reward_returned',reward_returned)
                # print(f"Next state",next_states)
                j_epsilon = reward_returned * perturbation
                total_j_epsilon += j_epsilon
                # print(j_epsilon)
            new_policy = current_policy + (Hyperparameters.alpha * (1/(Hyperparameters.sigma * Hyperparameters.num_perturbations)) * total_j_epsilon)
            current_policy = new_policy
            j_final.append(CartPole.estimate_j(current_policy))
        print(j_final)
        return j_final
        
    def estimate_j(policy):
        states = Model.initial_states
        for i in range(Hyperparameters.N):
            count = 0
            reward = 0
            while((states['x']>=-2.4 and states['x']<=2.4) and (states['omega']>=(-np.pi/15) and states['omega']<=(np.pi/15)) and (count<500)):
                # print('states',states)
                phi_s = CartPole.get_phi_s(states)
                #print('phi_s',phi_s)
                threshold = np.dot(phi_s.T, policy)
                #print('threshold',threshold)
                if threshold <= 0:
                    action = "left"
                else:
                    action = "right"
                next_state = CartPole.compute_next_states(states,action)
                states = next_state
                count += 1
                reward += 1
                # print(reward)
        return reward
    
    
    def compute_next_states(state,action):
        next_states = {}
        x,v,omega,omega_dot = state.values()
        force = Model.action_force[action]
        b = (force + (ModelDynamics.pole_mass * ModelDynamics.pole_length * (omega_dot**2) * math.sin(omega)))/ModelDynamics.total_mass
        c = ((ModelDynamics.gravity * math.sin(omega))- (b * math.cos(omega)))/(ModelDynamics.pole_length * (4/3 - (ModelDynamics.pole_mass * math.cos(omega)**2/ModelDynamics.total_mass)))
        d = b - (ModelDynamics.pole_mass * ModelDynamics.pole_length * c * math.cos(omega)/ModelDynamics.total_mass)
        next_states['x'] = x + (ModelDynamics.action_interval * v)
        next_states['v'] = v + (ModelDynamics.action_interval * d)
        next_states['omega'] = omega + (ModelDynamics.action_interval * omega_dot)
        next_states['omega_dot'] = omega_dot + (ModelDynamics.action_interval * c)
        # print(f"Next states:",next_states)
        return next_states
    
    def get_phi_s(states):
        x,v,omega,omega_dot = states.values()
        x = (x+2.4)/4.8
        v = (v-3.5)/7
        omega = (omega+np.pi/15)/(2*np.pi/15)
        omega_dot = (omega_dot-3.5)/7
        cosine_components = [1]+[np.cos(np.pi*x*i) for i in range(1, Hyperparameters.M+1)]+\
        [np.cos(np.pi*v*i) for i in range(1, Hyperparameters.M+1)]+\
        [np.cos(np.pi*omega*i) for i in range(1, Hyperparameters.M+1)]+\
        [np.cos(np.pi*omega_dot*i) for i in range(1, Hyperparameters.M+1)]
        phi_s = np.array(cosine_components).T
        return phi_s
    
def main():
    j_est_list = []
    for i in range(20):
        j_estimate = CartPole.evolution_strategies()
        j_est_list.append(j_estimate)
    final_j_list = np.mean(j_est_list,axis=0)
    final_j_list_std = np.std(j_est_list,axis=0)
    plt.errorbar(range(len(final_j_list)),final_j_list,yerr=final_j_list_std,label='Standard Deviation')
    plt.xlabel('Policy(Theta)')
    plt.ylabel('Average return')
    plt.title('Policy updation vs Average return')
    plt.plot(final_j_list)
    plt.show()
    
    
        
if __name__ == "__main__":
        main()