from sawp.workflows.product_runner import run_product


def main() -> int:
    run_product("figure-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
