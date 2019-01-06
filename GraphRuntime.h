#ifndef __graph_runtime_h__
#define __graph_runtime_h__

#include <dmlc/json.h>
#include <tvm/runtime/ndarray.h>

using tvm::runtime::NDArray;

/*

  Here we borrow a partial implementation of GraphRuntime from here:

  tvm/src/runtime/graph/graph_runtime.h

  Since there doesn't seem to be a public hook to access the layer info
  that is required for logging layer output at runtime.  In particular,
  we need access to:

  1) the number of layers : needed to iterate over all layers
  2) the layer attributes : needed by to allocate DLTensor * if using debug_get_output

  This info is all contained in private member variables of GraphRuntime after
  loading the JSON graph, so we simply duplicate the parts we need here for
  the C++ test and serialization.

 */

struct GraphRuntimePrivateStuff
{
    /*! \brief operator attributes about tvm op */
    struct TVMOpParam
    {
        std::string func_name;
        uint32_t num_inputs;
        uint32_t num_outputs;
        uint32_t flatten_data;
    };

    struct PoolEntry
    {
        size_t size;
        int device_type;

        PoolEntry(int s, int dev_type)
            : size(s)
            , device_type(dev_type)
        {
        }
    };

    // Node entry
    struct NodeEntry
    {
        uint32_t node_id;
        uint32_t index;
        uint32_t version;

        // JSON Loader
        void Load(dmlc::JSONReader* reader)
        {
            reader->BeginArray();
            CHECK(reader->NextArrayItem()) << "invalid json format";
            reader->Read(&node_id);
            CHECK(reader->NextArrayItem()) << "invalid json format";
            reader->Read(&index);
            if (reader->NextArrayItem())
            {
                reader->Read(&version);
                CHECK(!reader->NextArrayItem()) << "invalid json format";
            }
            else
            {
                version = 0;
            }
        }
    };

    // Node
    struct Node
    {
        // operator type in string
        std::string op_type;
        // name of the op
        std::string name;
        // parameters
        TVMOpParam param;
        // inputs
        std::vector<NodeEntry> inputs;
        // control deps
        std::vector<uint32_t> control_deps;

        // JSON Loader
        void LoadAttrs(dmlc::JSONReader* reader, TVMOpParam* param)
        {
            int bitmask = 0;
            std::string key, value;
            reader->BeginObject();
            while (reader->NextObjectItem(&key))
            {
                reader->Read(&value);
                if (key == "func_name")
                {
                    param->func_name = value;
                    bitmask |= 1;
                }
                else if (key == "num_inputs")
                {
                    param->num_inputs = strtoul(value.c_str(), nullptr, 10);
                    bitmask |= 2;
                }
                else if (key == "num_outputs")
                {
                    param->num_outputs = strtoul(value.c_str(), nullptr, 10);
                    bitmask |= 4;
                }
                else if (key == "flatten_data")
                {
                    param->flatten_data = strtoul(value.c_str(), nullptr, 10);
                    bitmask |= 8;
                }
            }
            CHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "invalid format";
        }

        // JSON Loader
        void Load(dmlc::JSONReader* reader)
        {
            reader->BeginObject();
            int bitmask = 0;
            std::string key;
            while (reader->NextObjectItem(&key))
            {
                if (key == "op")
                {
                    reader->Read(&op_type);
                    bitmask |= 1;
                }
                else if (key == "name")
                {
                    reader->Read(&name);
                    bitmask |= 2;
                }
                else if (key == "inputs")
                {
                    reader->Read(&inputs);
                    bitmask |= 4;
                }
                else if (key == "attr" || key == "attrs")
                {
                    this->LoadAttrs(reader, &param);
                }
                else if (key == "control_deps")
                {
                    reader->Read(&control_deps);
                }
                else
                {
                    LOG(FATAL) << "do not support key " << key;
                }
            }
            CHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
        }
    };

    struct GraphAttr
    {
        size_t storage_num_not_alloctaed{ 0 };
        std::vector<int> storage_id;
        std::vector<int> device_index;
        std::vector<std::string> dltype;
        std::vector<std::vector<int64_t>> shape;

        // The graph attribute fields.
        void Load(dmlc::JSONReader* reader)
        {
            reader->BeginObject();
            int bitmask = 0;
            std::string key, type;
            while (reader->NextObjectItem(&key))
            {
                if (key == "dltype")
                {
                    reader->BeginArray();
                    CHECK(reader->NextArrayItem());
                    reader->Read(&type);
                    CHECK_EQ(type, "list_str");
                    CHECK(reader->NextArrayItem());
                    reader->Read(&dltype);
                    CHECK(!reader->NextArrayItem());
                    bitmask |= 1;
                }
                else if (key == "storage_id")
                {
                    reader->BeginArray();
                    CHECK(reader->NextArrayItem());
                    reader->Read(&type);
                    CHECK_EQ(type, "list_int");
                    CHECK(reader->NextArrayItem());
                    reader->Read(&storage_id);
                    CHECK(!reader->NextArrayItem());
                    bitmask |= 2;
                }
                else if (key == "shape")
                {
                    reader->BeginArray();
                    CHECK(reader->NextArrayItem());
                    reader->Read(&type);
                    CHECK_EQ(type, "list_shape");
                    CHECK(reader->NextArrayItem());
                    reader->Read(&shape);
                    CHECK(!reader->NextArrayItem());
                    bitmask |= 4;
                }
                else if (key == "device_index")
                {
                    reader->BeginArray();
                    CHECK(reader->NextArrayItem());
                    reader->Read(&type);
                    CHECK_EQ(type, "list_int");
                    CHECK(reader->NextArrayItem());
                    reader->Read(&device_index);
                    CHECK(!reader->NextArrayItem());
                }
                else
                {
                    reader->BeginArray();
                    CHECK(reader->NextArrayItem());
                    reader->Read(&type);
                    if (type == "list_int")
                    {
                        CHECK(reader->NextArrayItem());
                        std::vector<int> temp;
                        reader->Read(&temp);
                    }
                    else if (type == "size_t")
                    {
                        CHECK(reader->NextArrayItem());
                        size_t temp;
                        reader->Read(&temp);
                    }
                    else
                    {
                        LOG(FATAL) << "cannot skip graph attr " << key;
                    }
                    CHECK(!reader->NextArrayItem());
                }
            }
            CHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
        }
    };

    // The graph attribute fields.
    void Load(dmlc::JSONReader* reader)
    {
        reader->BeginObject();
        int bitmask = 0;
        std::string key;
        while (reader->NextObjectItem(&key))
        {
            if (key == "nodes")
            {
                reader->Read(&nodes_);
                bitmask |= 1;
            }
            else if (key == "arg_nodes")
            {
                reader->Read(&input_nodes_);
                bitmask |= 2;
            }
            else if (key == "node_row_ptr")
            {
                reader->Read(&node_row_ptr_);
                bitmask |= 4;
            }
            else if (key == "heads")
            {
                reader->Read(&outputs_);
                bitmask |= 8;
            }
            else if (key == "attrs")
            {
                reader->Read(&attrs_);
                bitmask |= 16;
            }
            else
            {
                LOG(FATAL) << "key " << key << " is not supported";
            }
        }
        CHECK_EQ(bitmask, 1 | 2 | 4 | 8 | 16) << "invalid format";
    }

    // Get node entry index.
    uint32_t entry_id(uint32_t nid, uint32_t index) const
    {
        return node_row_ptr_[nid] + index;
    }

    // Get node entry index.
    uint32_t entry_id(const NodeEntry& e) const
    {
        return entry_id(e.node_id, e.index);
    }

    // Number of node entries.
    uint32_t num_node_entries() const
    {
        return node_row_ptr_.back();
    }

    /*! \brief The graph nodes. */
    std::vector<Node> nodes_;
    /*! \brief The argument nodes. */
    std::vector<uint32_t> input_nodes_;
    /*! \brief Used for quick entry indexing. */
    std::vector<uint32_t> node_row_ptr_;
    /*! \brief Output entries. */
    std::vector<NodeEntry> outputs_;
    /*! \brief Additional graph attributes. */
    GraphAttr attrs_;
    /*! \brief The code module that contains both host and device code. */
    tvm::runtime::Module module_;
    /*! \brief Execution context of all devices including the host. */
    std::vector<TVMContext> ctxs_;
    /*! \brief Common storage pool for all devices. */
    std::vector<NDArray> storage_pool_;
    /*! \brief Data entry of each node. */
    std::vector<NDArray> data_entry_;
};

#endif // __graph_runtime_h__
