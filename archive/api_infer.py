from fiboaisdk.api_aisdk_py import api_infer_py
import json
import numpy as np


class InferenceSession:
    """
    InferenceSession handles model loading and inference execution on various platforms.

    Args:
        model (str): Path to the model file.
        platform (str): Target platform (e.g., 'qualcomm', 'axera', 'cpu').
        framework (str): Inference framework used (e.g., 'snpe', 'qnn', 'mnn').
        runtime (str): Runtime backend (e.g., 'trace', 'debug', 'info', 'warn', 'error', 'critical').
        log_level (str): Logging level (e.g., 'CPU', 'GPU', 'DSP', 'NPU').
        profile_level (int): Performance profile index.
            - For SNPE: 0~9, 0=BALANCED, 1=HIGH_PERFORMANCE, 2=POWER_SAVER, 3=SYSTEM_SETTINGS, 4=SUSTAINED_HIGH_PERFORMANCE, 5=BURST, 6=LOW_POWER_SAVER, 7=HIGH_POWER_SAVER, 8=LOW_BALANCED, 9=EXTREME_POWERSAVER
            - For QNN: 0~13, 0=LOW_BALANCED, 1=BALANCED, 2=DEFAULT, 3=HIGH_PERFORMANCE, 4=SUSTAINED_HIGH_PERFORMANCE, 5=BURST, 6=EXTREME_POWER_SAVER, 7=LOW_POWER_SAVER, 8=POWER_SAVER, 9=HIGH_POWER_SAVER, 10=SYSTEM_SETTINGS, 11=NO_USER_INPUT, 12=CUSTOM, 13=INVALID
            - For AXERA: Currently only 0=NOT_SUPPORTED.
    """
    def __init__(self, model: str = "None",
                    platform: str = "None",
                    framework: str = "None",
                    runtime: str = "CPU",
                    log_level : str = "ERROR",
                    profile_level : int = 5 # burst mode for snpe and qnn
                    ):
        """
        Description:
            Encapsulates all model-related parameters required for running inference.
            Upon instantiation, buffer allocation, performance level, and runtime environment are configured.

            Runtime selection priority:
                - CPU (default)
                - GPU (if specified)
                - DSP (if specified)
                - NPU (if specified)

            Note:
                If using DSP as the runtime, SNPE Hexagon libraries must be pushed to the working directory.

        Args:
            model (str): Full path to the model on the device, including the file name 
                         (e.g., '/data/local/tmp/models/yolonas/Quant_yoloNas_s_320.dlc').

            platform (str): Target hardware platform. Supported values: 'QUALCOMM', 'AXERA'. Defaults to "None".

            framework (str): Inference framework to use. Supported values: 'SNPE', 'QNN', 'AXERA'. Defaults to "None".

            runtime (str): Target runtime environment. Supported values: 'CPU', 'GPU', 'DSP', 'NPU'. Defaults to 'CPU'.

            log_level (str): Logging verbosity level. Defaults to 'ERROR'.

            profile_level (str): Performance profile level. Defaults to 'BURST'.
        """
        self.m_model = model
        self.m_platform = platform
        self.m_framework = framework
        self.m_runtime = runtime
        self.log_level = log_level
        self.profiling_level = profile_level
        self.m_params = api_infer_py.InferParams(self.m_model, self.m_platform, self.m_framework, self.m_runtime, self.log_level, self.profiling_level)
        self.m_session = api_infer_py.InferAPI()
    

    def Initialize(self):
        """
        Description:
            Initializes buffers and loads the network.
        Returns:
            Framework-specific context or session object after initialization.
        """
        return self.m_session.Init(self.m_params)
    

    def Execute(self, output_names, input_feed):
        """
        Description:
            Executes inference using the provided input data.

        Args:
            output_names (list): List of output tensor names to fetch after inference.
            input_feed (dict): Dictionary mapping input tensor names to data. Data can be 
                               NumPy arrays or Python lists.

        Returns:
            dict or None: Returns a dictionary of output tensors if execution is successful;
                          otherwise, returns None.

        Raises:
            ValueError: If input_feed is empty.
            TypeError: If input type is unsupported.
            ValueError: If the framework is SNPE and the input type is not float32.
        """
        if not input_feed:
            raise ValueError("input_feed is empty")
    
        # judge input tensor's data type
        first_value = next(iter(input_feed.values()))
    
        if isinstance(first_value, np.ndarray):
            dtype = first_value.dtype
            type_name = dtype.name  # gives 'float32', 'int64', etc.
            input_feed = {k: v.flatten().tolist() for k, v in input_feed.items()}

        elif isinstance(first_value, list):
            if not first_value:
                raise ValueError("Input list is empty")

            elem = first_value[0]

            if isinstance(elem, float):
                type_name = 'float32'  # default float32
            elif isinstance(elem, int):
                min_val = min(first_value)
                max_val = max(first_value)

                if min_val >= 0:
                    if max_val <= np.iinfo(np.uint8).max:
                        type_name = 'uint8'
                    elif max_val <= np.iinfo(np.uint16).max:
                        type_name = 'uint16'
                    elif max_val <= np.iinfo(np.uint32).max:
                        type_name = 'uint32'
                    else:
                        type_name = 'uint64'
                else:
                    if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                        type_name = 'int8'
                    elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                        type_name = 'int16'
                    elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                        type_name = 'int32'
                    else:
                        type_name = 'int64'
            else:
                raise TypeError("Unsupported element type in list")

        else:
            raise TypeError(f"Unsupported input type: {type(first_value)}")
    
        print(f"[INFO] input type: {type_name}")
        
        # SNPE only support float32
        if self.m_framework.lower() == "snpe" and type_name != 'float32':
            raise ValueError("Snpe only support float32 data type")
    
        dispatch_table = {
            'float32': ('Execute_float', 'FetchOutputs_float'),
            'float': ('Execute_float', 'FetchOutputs_float'),
            'int32': ('Execute_int32', 'FetchOutputs_int32'),
            'uint32': ('Execute_uint32', 'FetchOutputs_uint32'),
            'int16': ('Execute_int16', 'FetchOutputs_int16'),
            'uint16': ('Execute_uint16', 'FetchOutputs_uint16'),
            'int8': ('Execute_int8', 'FetchOutputs_int8'),
            'uint8': ('Execute_uint8', 'FetchOutputs_uint8'),
            'int64': ('Execute_int64', 'FetchOutputs_int64'),
            'uint64': ('Execute_uint64', 'FetchOutputs_uint64'),
            'np.float32': ('Execute_float', 'FetchOutputs_float'),
            'np.int32': ('Execute_int32', 'FetchOutputs_int32'),
            'np.uint32': ('Execute_uint32', 'FetchOutputs_uint32'),
            'np.int16': ('Execute_int16', 'FetchOutputs_int16'),
            'np.uint16': ('Execute_uint16', 'FetchOutputs_uint16'),
            'np.int8': ('Execute_int8', 'FetchOutputs_int8'),
            'np.uint8': ('Execute_uint8', 'FetchOutputs_uint8'),
            'np.int64': ('Execute_int64', 'FetchOutputs_int64'),
            'np.uint64': ('Execute_uint64', 'FetchOutputs_uint64'),
        }
    
        if type_name not in dispatch_table:
            raise TypeError(f"Unsupported input dtype: {type_name}")
    
        exec_func_name, fetch_func_name = dispatch_table[type_name]
        exec_func = getattr(self.m_session, exec_func_name)
        fetch_func = getattr(self.m_session, fetch_func_name)
    
        success = exec_func(input_feed)
        return fetch_func(output_names) if success == 0 else None


    def Release(self):
        """
        Description:
            Releases all allocated resources associated with the session.
        
        Returns:
            int: Return code from the underlying release function.
        """
        return self.m_session.Release()


class OnnxContext:
    """
    Description:
        ONNX模型推理上下文，参考SnpeContext实现。
    Args:
        onnx_path : ONNX模型路径
        output_tensors : 输出tensor名称列表
        runtime : 运行后端，默认CPU
        log_level : 日志等级，默认INFO
    """
    def __init__(self, onnx_path: str = "None",
                 output_tensors: list = [],
                 runtime: str = "CPU",
                 log_level: str = "INFO"):
        self.m_onnxpath = onnx_path
        self.m_output_tensors = output_tensors
        self.m_runtime = runtime
        self.log_level = log_level
        self.m_context = api_infer_py.InferAPI()

    def Initialize(self):
        user_values = {
            "logger": {
                "log_level": self.log_level,
            },
            "all_models": [
                {
                    "model_name": self.m_onnxpath,
                    "model_path": self.m_onnxpath,
                    "run_framework": "onnx",
                    "run_backend": self.m_runtime,
                    "output_names": self.m_output_tensors,
                    "external_params": {}
                }
            ],
            "graphs": [
                {
                    "all_nodes_params": {
                        "nodes": [
                            {
                                "model_name": self.m_onnxpath,
                                "run_framework": "onnx",
                                "run_backend": self.m_runtime,
                            }
                        ]
                    }
                }
            ]
        }
        return self.m_context.Init(generate_config(user_values))

    def Execute(self, output_names, input_feed):
        input_feed = {k: v.astype(np.float32).flatten().tolist() for k, v in input_feed.items()}
        if self.m_context.Execute(input_feed) == 0:
            return self.m_context.FetchOutputs(output_names)
        else:
            return None

    def Release(self):
        return self.m_context.Release()
        