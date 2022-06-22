package com.github.hiaac;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.security.InvalidKeyException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Java Environment API for the ML-Agents Toolkit
 * The aim of this API is to expose Agents evolving in a simulation
 * to perform reinforcement learning on.
 * This API supports multi-agent scenarios and groups similar Agents (same
 * observations, actions spaces and behavior) together. These groups of Agents are
 * identified by their BehaviorName.
 * For performance reasons, the data of each group of agents is processed in a
 * batched manner. Agents are identified by a unique AgentId identifier that
 * allows tracking of Agents across simulation steps. Note that there is no
 * guarantee that the number or order of the Agents in the state will be
 * consistent across simulation steps.
 * A simulation steps corresponds to moving the simulation forward until at least
 * one agent in the simulation sends its observations to Java again. Since
 * Agents can request decisions at different frequencies, a simulation step does
 * not necessarily correspond to a fixed simulation time increment.
 **/
public class BaseEnv {
    /**
     * Contains the data a single Agent collected since the last
     * simulation step.
     * <p>
     * - obs is a list of nd4j arrays observations collected by the agent.
     * - reward is a float. Corresponds to the rewards collected by the agent
     * since the last simulation step.
     * - agentId is an int and a unique identifier for the corresponding Agent.
     * - actionMask is an optional list of one dimensional array of booleans.
     * Only available when using multi-discrete actions.
     * Each array corresponds to an action branch. Each array contains a mask
     * for each action of the branch. If true, the action is not available for
     * the agent during this simulation step.
     **/
    public static class DecisionStep {
        public ArrayList<INDArray> obs;
        public float reward;
        public int agentId;
        public ArrayList<INDArray> actionMask;
        public int groupId;
        public float groupReward;
    }

    /**
     * Contains the data a batch of similar Agents collected since the last
     * simulation step. Note that all Agents do not necessarily have new
     * information to send at each simulation step. Therefore, the ordering of
     * agents and the batch size of the DecisionSteps are not fixed across
     * simulation steps.
     * <p>
     * - obs is a list of nd4j arrays observations collected by the batch of
     * agent. Each obs has one extra dimension compared to DecisionStep: the
     * first dimension of the array corresponds to the batch size of the batch.
     * - reward is a float vector of length batch size. Corresponds to the
     * rewards collected by each agent since the last simulation step.
     * - agentId is an int vector of length batch size containing unique
     * identifier for the corresponding Agent. This is used to track Agents
     * across simulation steps.
     * - actionMask is an optional list of two-dimensional array of booleans.
     * Only available when using multi-discrete actions.
     * Each array corresponds to an action branch. The first dimension of each
     * array is the batch size and the second contains a mask for each action of
     * the branch. If true, the action is not available for the agent during
     * this simulation step.
     **/
    public class DecisionSteps {
        public ArrayList<INDArray> obs;
        public INDArray reward;
        public INDArray agentId;
        public INDArray action_mask;
        public INDArray groupId;
        public INDArray groupReward;
        public HashMap<Integer, Integer> agentIdToIndex;

        /**
         * Returns an empty DecisionSteps.
         *
         * @param spec The BehaviorSpec for the DecisionSteps
         **/
        public DecisionSteps(BehaviorSpec spec) {
            this.obs = new ArrayList<>();
            for (var sen_spec : spec.observationSpecs) {
                var aux = new long[sen_spec.shape.length + 1];
                System.arraycopy(sen_spec.shape, 0, aux, 1, sen_spec.shape.length);
                aux[0] = 0;
                this.obs.add(Nd4j.zeros(DataType.FLOAT, aux));
            }
            this.reward = Nd4j.empty(DataType.FLOAT);
            this.agentId = Nd4j.empty(DataType.INT32);
            this.action_mask = null;
            this.groupId = Nd4j.empty(DataType.INT32);
            this.groupReward = Nd4j.empty(DataType.FLOAT);
            this.agentIdToIndex = null;
        }

        /**
         * @return A Dict that maps agent_id to the index of those agents in
         * this DecisionSteps.
         **/
        public HashMap<Integer, Integer> agentIdToIndex() {
            if (this.agentIdToIndex == null) {
                this.agentIdToIndex = new HashMap<>();
                for (int i = 0; i < this.agentId.columns(); i++)
                    this.agentIdToIndex.put(this.agentId.getInt(i), i);
            }
            return this.agentIdToIndex;
        }

        public int len() {
            return this.agentId.columns();
        }

        /**
         * returns the DecisionStep for a specific agent.
         *
         * @param agent_id The id of the agent
         * @return The DecisionStep
         **/
        public DecisionStep getItem(int agent_id) throws InvalidKeyException {
            if (!this.agentIdToIndex.containsKey(agent_id))
                throw new InvalidKeyException("agent_id " + agent_id + " is not present in the DecisionSteps");
            int agent_index = this.agentIdToIndex.get(agent_id);
            DecisionStep ret = new DecisionStep();
            ret.obs = new ArrayList<>();
            for (var batched_obs : this.obs)
                ret.obs.add(batched_obs.getRow(agent_index));
            ret.actionMask = null;
            if (this.action_mask != null) {
                ret.actionMask = new ArrayList<>();
                for (int i = 0; i < this.action_mask.columns(); i++)
                    ret.actionMask.add(this.action_mask.getRow(agent_index));
            }
            ret.groupId = groupId.getInt(agent_index);
            ret.reward = this.reward.getFloat(agent_index);
            ret.agentId = agent_id;
            ret.groupReward = this.groupReward.getFloat(agent_index);
            return ret;
        }
    }

    public class ActionSpec {
    }

    public class DimensionProperty {
    }

    /**
     * An Enum which defines the type of information carried in the observation
     * of the agent.
     **/
    public enum ObservationType {
        DEFAULT, // Observation information is generic.
        GOAL_SIGNAL // Observation contains goal information for current task.
    }

    /**
     * A class containing information about the observation of Agents.
     * <p>
     * - shape is an array of int: It corresponds to the shape of
     * an observation's dimensions.
     * - dimensionProperty is an array of DimensionProperties flag, one flag for each
     * dimension.
     * - observationType is an enum of ObservationType.
     **/
    public class ObservationSpec {
        public long[] shape;
        public DimensionProperty[] dimensionProperty;
        public ObservationType observationType;
        public String name = null; // Optional name. For observations coming from com.unity.ml-agents, this will be the ISensor name.
    }

    /**
     * A class containing information about the observation and action
     * spaces for a group of Agents under the same behavior.
     * <p>
     * - observationSpecs is a List of ObservationSpec object containing
     * information about the information of the Agent's observations such as their shapes.
     * The order of the ObservationSpec is the same as the order of the observations of an
     * agent.
     * - actionSpec is an ActionSpec object.
     **/
    public class BehaviorSpec {
        ArrayList<ObservationSpec> observationSpecs;
        ActionSpec actionSpec;
    }
}
