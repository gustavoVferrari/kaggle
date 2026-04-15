import json

def to_jsonl(data, file_path=None, ensure_ascii=False, mode="write"):
    """
    Converte dict, list ou pandas.DataFrame para JSONL.

    :param data: dict, list ou pandas.DataFrame
    :param file_path: caminho do arquivo .jsonl (opcional)
    :param ensure_ascii: False mantém acentos corretamente
    :param mode: "write" sobrescreve o arquivo | "append" acrescenta linhas ao final
    :return: string JSONL (se file_path não for informado)
    """
    if mode not in ("write", "append"):
        raise ValueError("O parâmetro 'mode' deve ser 'write' ou 'append'.")

    lines = []

    # Tipos simples → lista
    if isinstance(data, (int, float, str, bool)):
        data = [data]

    # Importação tardia para não exigir pandas se não for usado
    try:
        import pandas as pd
        is_dataframe = isinstance(data, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    if isinstance(data, dict):
        lines.append(json.dumps(data, ensure_ascii=ensure_ascii))

    elif isinstance(data, list):
        for item in data:
            lines.append(json.dumps(item, ensure_ascii=ensure_ascii))

    elif is_dataframe:
        for _, row in data.iterrows():
            lines.append(json.dumps(row.to_dict(), ensure_ascii=ensure_ascii))

    else:
        raise TypeError("Tipo não suportado. Use dict, list ou pandas.DataFrame.")

    jsonl_content = "\n".join(lines)

    if file_path:
        file_mode = "a" if mode == "append" else "w"
        with open(file_path, file_mode, encoding="utf-8") as f:
            # Em modo append, garante que a nova linha começa em uma linha nova
            if mode == "append":
                f.seek(0, 2)  # Move o cursor para o final do arquivo
                if f.tell() > 0:  # Se o arquivo não estiver vazio, adiciona \n antes
                    f.write("\n")
            f.write(jsonl_content)
        return None

    return jsonl_content
