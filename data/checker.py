# -*-coding: utf-8 -*-
import json
import logging

from jsonschema import validate

class Checker(object):
    @staticmethod
    def check_roles(conversation):
        ret = []
        for i, msg in enumerate(conversation):
            if msg["role"] == "system":
                assert i == 0
            elif msg["role"] == "user":
                assert i == 0 or i == 1 or (
                            ret[-1]["role"] in ("system", "assistant", "tool") and not ret[-1].get("tool_calls")), f"user error, idx {i} {ret[-1] if ret else None}"
            elif msg["role"] == "assistant" and not msg.get("tool_calls"):
                assert ret[-1]["role"] in ("system", "user", "tool"), f"assistant content error {ret[-1]} {msg}"
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                assert ret[-1]["role"] == "user" or ret[-1]["role"] == "tool", "assistant functioncall error"
            elif msg["role"] == "tool":
                assert (ret[-1]["role"] == "assistant" and ret[-1].get("tool_calls") or ret[-1]["role"] == "tool"), "function error"
            ret.append(msg)

    @staticmethod
    def check_basic(data):
        assert "conversation" in data and type(data["conversation"]) is list
        assert "tools" in data and (data["tools"] is None or type(data["tools"]) is str)

        for msg in data["conversation"]:
            if msg["role"] in ("system", "user"):
                assert msg["content"] and not msg.get("tool_calls")
            elif msg["role"] == "assistant":
                assert msg.get("content") or msg.get("tool_calls"), msg
            elif msg["role"] == "tool":
                assert msg["content"], msg
            else:
                raise AssertionError("error role")

    @staticmethod
    def check_schema(data):
        # TODO 考虑required字段
        conversation = data["conversation"]
        functions = [t["function"] for t in json.loads(data["tools"])]
        name_to_schema = {f["name"]: f["parameters"] for f in functions}
        for msg in conversation:
            if msg["role"] == "assistant" and (funcs := msg.get("tool_calls")):
                for func in funcs:
                    func_name = func["function"]["name"]
                    arguments = json.loads(func["function"]["arguments"])
                    schema = name_to_schema[func_name]

                    try:
                        validate(instance=arguments, schema=schema)
                    except Exception as e:
                        print(arguments, schema)
                        # logging.exception(e)
                        # print(f"{func_name} schema error!")
                        raise e

    def check(self, data):
        try:
            self.check_basic(data)
        except Exception as e:
            # logging.exception(e)
            return "basic_error"

        try:
            self.check_roles(data["conversation"])
        except Exception as e:
            # logging.exception(e)
            return "roles_error"

        try:
            self.check_schema(data)
        except Exception as e:
            logging.exception(e)
            return "schema_error"
        return None
