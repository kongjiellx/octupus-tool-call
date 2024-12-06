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
                            ret[-1]["role"] in ("system", "assistant", "function") and not ret[-1].get("function_call")), f"user error, idx {i} {ret[-1] if ret else None}"
            elif msg["role"] == "assistant" and not msg.get("function_call"):
                assert ret[-1]["role"] in ("user", "function"), f"assistant content error {ret[-1]} {msg}"
            elif msg["role"] == "assistant" and msg.get("function_call"):
                assert ret[-1]["role"] == "user" or ret[-1]["role"] == "function", "assistant functioncall error"
            elif msg["role"] == "function":
                assert ret[-1]["role"] == "assistant" and ret[-1].get("function_call"), "function error"
            ret.append(msg)

    @staticmethod
    def check_basic(data):
        assert "conversation" in data and type(data["conversation"]) is list
        assert "functions" in data and (data["functions"] is None or type(data["functions"]) is str)

        for msg in data["conversation"]:
            if msg["role"] in ("system", "user"):
                assert msg["content"] and not msg.get("function_call")
            elif msg["role"] == "assistant":
                assert msg.get("content") or msg.get("function_call"), msg
            elif msg["role"] == "function":
                assert msg["content"], msg
            else:
                raise AssertionError("error role")

    @staticmethod
    def check_schema(data):
        # TODO 考虑required字段
        conversation = data["conversation"]
        functions = data["functions"]
        name_to_schema = {f["name"]: f["parameters"] for f in json.loads(functions)}
        for msg in conversation:
            if msg["role"] == "assistant" and (funcs := msg.get("function_call")):
                for func in json.loads(funcs):
                    func_name = func["name"]
                    arguments = func["arguments"]
                    schema = name_to_schema[func_name]

                    try:
                        validate(instance=arguments, schema=schema)
                    except Exception as e:
                        # print(arguments, schema)
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
            # logging.exception(e)
            return "schema_error"
        return None
