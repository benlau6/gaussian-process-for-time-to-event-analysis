from typing import Awaitable, Any
import asyncio
from pathlib import Path


async def run_sequence(*funcs: Awaitable[Any]) -> Any:
    return [await func for func in funcs]


async def run_parallel(*funcs: Awaitable[Any]) -> Any:
    return await asyncio.gather(*funcs)


def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent


def get_package_root() -> Path:
    return Path(__file__).absolute().parent


def get_package_name() -> str:
    return Path(__file__).parent.name


def get_project_name() -> str:
    return Path(__file__).parent.parent.name


def main():
    print(
        f"{get_project_name()=}\n"
        f"{get_package_name()=}\n"
        f"{get_project_root()=}\n"
        f"{get_package_root()=}\n"
    )


if __name__ == "__main__":
    main()
