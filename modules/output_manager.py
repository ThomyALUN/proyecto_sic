# output_manager.py

import pandas as pd

def export_solution_to_excel(best_solution,
                             filename,
                             group_subject,
                             group_names,
                             group_number,
                             asignaturas,
                             docentes,
                             docente_ids,
                             D):
    rows = []
    for i, gene in enumerate(best_solution):
        # Identificar asignatura (por su ID) y su nombre
        aid = group_subject[i]
        asignatura_nombre = asignaturas[aid]["Nombre"]
        
        # Número de grupo (1, 2, 3...)
        numero_grupo = group_number[i]
        
        # Docente asignado
        if gene < D:
            docente_id = docente_ids[gene]
            docente_nombre = docentes[docente_id]["Nombre"]
        else:
            docente_nombre = "Docente Ocasional"
        
        rows.append({
            "Asignatura": asignatura_nombre,
            "Grupo": numero_grupo,
            "Docente asignado": docente_nombre
        })
    
    # DataFrame y exportación
    df_resultado = pd.DataFrame(rows, columns=["Asignatura", "Grupo", "Docente asignado"])
    df_resultado.to_excel(filename, index=False)
    print(f"Archivo '{filename}' generado con éxito.")


def print_final_solution(best_solution,
                         group_subject,
                         group_names,
                         asignaturas,
                         docente_ids,
                         docentes,
                         D):
    print("\nAsignación final por grupo:")
    for i, gene in enumerate(best_solution):
        aid = group_subject[i]
        asignatura_nombre = asignaturas[aid]["Nombre"]
        grupo_nombre = group_names[i]
        
        if gene < D:
            docente_id = docente_ids[gene]
            docente_nombre = docentes[docente_id]["Nombre"]
        else:
            docente_nombre = "Docente Ocasional"
        
        print(f"Grupo: {grupo_nombre} (Asignatura: {asignatura_nombre}) --> {docente_nombre}")
