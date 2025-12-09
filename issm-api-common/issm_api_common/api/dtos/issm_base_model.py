import json
from collections import defaultdict
from datetime import datetime
from enum import Enum
from hashlib import sha512
from json import JSONDecodeError
from pathlib import Path
from typing import Any, TypeVar, Type, Union, Optional, Dict, Tuple, List

from pydantic import BaseModel, Extra, ValidationError

from issm_api_common.api.logger import logger


def to_camel_case(st: str):
    output = "".join(x for x in st.title() if x.isalnum())
    return output[0].lower() + output[1:]


TIssmPydanticModel = TypeVar("TIssmPydanticModel", bound="IssmPydanticModel")


# ruff: noqa: C901
def _get_attribute_path_for_all_attributes(
    obj: Any,
    path: List[Any],
    collection_of_paths: List[Any],
    only_attributes_starting_with: Optional[Union[str, List[str]]] = None,
    _object_from_attribute: bool = False,
):
    def _should_continue_to_recurse(path_element: Any):
        if only_attributes_starting_with is not None:
            if isinstance(path_element, str) and path_element.startswith(
                tuple(only_attributes_starting_with)
            ):
                return False
            else:
                return True
        else:
            return True

    obj_is_instance_of_class = False
    if (obj is not None) and (not isinstance(obj, (str, float, int, list, dict, set))):
        try:
            vars(obj)
            obj_is_instance_of_class = True
        except TypeError:  # can happen for instance in the case of numpy arrays
            obj_is_instance_of_class = False

    if obj_is_instance_of_class:
        for attr, _ in vars(obj).items():
            if attr.startswith("__"):
                # we don't want to traverse internal objects (__*)
                continue
            temp_path = path[:]
            temp_path.append(attr)
            o = getattr(obj, attr)
            if not _should_continue_to_recurse(attr):
                o = "recursion_stopped"
            _get_attribute_path_for_all_attributes(
                o, temp_path, collection_of_paths, only_attributes_starting_with, True
            )
    elif (obj is not None) and (isinstance(obj, list)):
        for i, o in enumerate(obj):
            temp_path = path[:]
            temp_path.append([i])
            _get_attribute_path_for_all_attributes(
                o, temp_path, collection_of_paths, only_attributes_starting_with, False
            )
    elif (obj is not None) and (isinstance(obj, set)):
        for i, o in enumerate(obj):
            temp_path = path[:]
            temp_path.append({i})
            _get_attribute_path_for_all_attributes(
                o, temp_path, collection_of_paths, only_attributes_starting_with, False
            )
    elif (obj is not None) and (isinstance(obj, dict)):
        for k, v in obj.items():
            temp_path = path[:]
            temp_path.append({k: k})
            _get_attribute_path_for_all_attributes(
                v, temp_path, collection_of_paths, only_attributes_starting_with, False
            )
    else:
        if only_attributes_starting_with is not None:
            if _object_from_attribute:
                if isinstance(path[-1], str) and path[-1].startswith(
                    tuple(only_attributes_starting_with)
                ):
                    collection_of_paths.append(path)
        else:
            collection_of_paths.append(path)


def get_attribute_path_for_all_attributes(
    obj: Any,
    only_attributes_starting_with: Optional[Union[str, List[str]]] = None,
    in_pydantic_exclusion_format: bool = False,
) -> Union[List[Any], Dict[str, Any]]:
    collection_of_paths = []
    _get_attribute_path_for_all_attributes(
        obj, [], collection_of_paths, only_attributes_starting_with
    )

    # python is pretty cool, see https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)

    # https://stackoverflow.com/questions/20428636/python-convert-defaultdict-to-dict
    def recursive_defaultdict2dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = recursive_defaultdict2dict(v)
        return dict(d)

    exclusion_dict = recursive_defaultdict()
    if in_pydantic_exclusion_format:
        for path in collection_of_paths:
            current_dict = {}
            for i, path_element in enumerate(path):
                if i == 0:
                    if i == len(path) - 1:
                        exclusion_dict[path_element] = ...
                    current_dict = exclusion_dict[path_element]
                else:
                    dict_element = None
                    if isinstance(path_element, str):
                        dict_element = path_element
                    if isinstance(path_element, set):
                        dict_element = path_element.pop()
                    if isinstance(path_element, list):
                        dict_element = path_element[0]
                    if isinstance(path_element, dict):
                        dict_element = list(path_element.keys())[0]
                    if i == len(path) - 1:
                        current_dict[dict_element] = ...
                    current_dict = current_dict[dict_element]
        return recursive_defaultdict2dict(exclusion_dict)
    return collection_of_paths


class IssmPydanticModel(BaseModel):
    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True
        populate_by_name = True
        use_enum_values = True
        alias_generator = to_camel_case

    @classmethod
    def _construct_with_kwargs_dict(cls, kwargs: Dict[str, Any]) -> "IssmPydanticModel":
        # set acts by default on keys
        kwargs_set = set(kwargs)
        # python short circuits
        try:
            if not kwargs["skip_kwargs_completeness_check"] and kwargs_set != set(
                cls.model_fields
            ):  # set acts on keys
                raise ValueError("Did not supply values for all field defaults")
        except KeyError:
            pass
        m = cls.__new__(cls)
        object.__setattr__(m, "__dict__", kwargs)
        object.__setattr__(m, "__fields_set__", kwargs_set)
        return m

    @classmethod
    def constr(cls, **kwargs) -> "IssmPydanticModel":
        return cls._construct_with_kwargs_dict(kwargs)

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise Exception(f'Key "{key}" doesn\'t exist in {self.__class__}')

    def __contains__(self, key: str):
        return key in self.__dict__

    def __setitem__(self, key: str, item: Any):
        setattr(self, key, item)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IssmPydanticModel):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __hash__(self) -> int:
        return int(
            sha512(
                json.dumps(self.as_dict(), ensure_ascii=True, sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest(),
            16,
        )

    def equals_ignoring(
        self,
        other: "IssmPydanticModel",
        ignore_fields_starting_with: Tuple[str, ...] = (),
    ) -> bool:
        return self.as_dict(
            ignore_fields_starting_with=ignore_fields_starting_with
        ) == other.as_dict(ignore_fields_starting_with=ignore_fields_starting_with)

    def to_json_string(
        self,
        ignore_fields_starting_with: Tuple[str, ...] = ("ij_",),
        ensure_ascii: bool = False,
        by_alias: bool = False,
        sort_keys: bool = True,
        indent: Optional[int] = 4,
        exclude_none: Optional[bool] = True,
        exclude_unset: bool = False,
        **kwargs,
    ):
        def known_data_formats_handler(obj: Any):
            if hasattr(obj, "jsonable"):
                # noinspection PyCallingNonCallable
                return obj.jsonable()
            if hasattr(obj, "json"):
                # noinspection PyCallingNonCallable
                return json.loads(
                    obj.json(
                        ensure_ascii=ensure_ascii,
                        encoder=known_data_formats_handler,
                        by_alias=by_alias,
                        exclude_unset=exclude_unset,
                        **kwargs,
                    )
                )
            if isinstance(obj, Path) or isinstance(obj, datetime):
                return str(obj)
            if isinstance(obj, Enum):
                return str(obj.value)
            if isinstance(obj, set):
                return list(obj)
            else:
                raise TypeError

        exclude_dict = get_attribute_path_for_all_attributes(
            self,
            # if private attributes are set via object.__setattr__ directly pydantic would normally not ignore them
            # during serialization. We however want to ignore them which is why we add '_' to
            # get_attribute_path_for_all_attributes.
            only_attributes_starting_with=list(ignore_fields_starting_with) + ["_"],
            in_pydantic_exclusion_format=True,
        )

        json_string = self.model_dump_json(
            by_alias=by_alias,
            exclude=exclude_dict,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            **kwargs,
        )
        return json.dumps(
            json.loads(json_string),
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
        )

    def to_json_file(self, json_file_path: Union[str, Path], **kwargs):
        json_file_path = Path(json_file_path)
        json_file_path.parent.mkdir(exist_ok=True, parents=True)
        with json_file_path.open("w", encoding="utf-8") as text_file:
            text_file.write(self.to_json_string(**kwargs))

    def as_dict(
        self,
        ignore_fields_starting_with: Tuple[str, ...] = (),
        exclude_none: Optional[bool] = False,
        exclude_unset: bool = False,
    ) -> Dict[str, Any]:
        return json.loads(
            self.to_json_string(
                ignore_fields_starting_with=ignore_fields_starting_with,
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
            )
        )

    @classmethod
    def from_json_string(cls: Type[TIssmPydanticModel], s: str) -> "TIssmPydanticModel":
        # noinspection PyArgumentList
        return cls(**json.loads(s))

    @classmethod
    def from_json_file(
        cls: Type[TIssmPydanticModel], json_file_path: Union[str, Path]
    ) -> "TIssmPydanticModel":
        if Path(json_file_path).exists():
            try:
                with Path(json_file_path).open("r", encoding="utf-8") as f:
                    # noinspection PyArgumentList
                    return cls(**json.load(f))
            except (UnicodeDecodeError, JSONDecodeError, ValidationError) as e:
                error_message = f"Found error with file at path = {json_file_path}"
                if hasattr(e, "message"):
                    e.message += f"\n{error_message}"
                else:
                    logger.error(error_message)
                raise e
        else:
            raise RuntimeError(f"Json file {json_file_path} does not exist.")
