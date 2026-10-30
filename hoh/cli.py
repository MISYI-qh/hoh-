import argparse

COMMANDS = ["train", "generate", "evaluate", "grow", "learn-tokenizer", "migrate-vocab", "lifelong", "chat", "learn", "experience", "reasoning-data", "goals", "service"]


def main(argv=None):
    p = argparse.ArgumentParser(prog="hoh", description="HOH language model tools")
    p.add_argument("command", choices=COMMANDS)
    args, rest = p.parse_known_args(argv)
    if args.command == "train":
        from . import train
        return train.train(rest)
    if args.command == "generate":
        from . import generate
        generate.main(rest)
    elif args.command == "evaluate":
        from . import evaluate
        evaluate.main(rest)
    elif args.command == "grow":
        from . import growth
        growth.main(rest)
    elif args.command == "service":
        from . import service
        return service.main(rest)
    elif args.command == "lifelong":
        from . import lifelong_cli
        return lifelong_cli.main(rest)
    elif args.command in ("chat", "learn", "experience", "reasoning-data", "goals"):
        from . import chat_cli
        return {"chat": chat_cli.chat_main, "learn": chat_cli.learn_main, "experience": chat_cli.experience_main,
                "reasoning-data": chat_cli.reasoning_data_main, "goals": chat_cli.goals_main}[args.command](rest)
    elif args.command == "learn-tokenizer":
        from . import vocab_cli
        vocab_cli.learn_main(rest)
    else:
        from . import vocab_cli
        vocab_cli.migrate_main(rest)
    return 0
