import os
from src.training.random_model import RandomModel

class AngleHelper:
  def __init__(self, *args, **kwargs):
    # TO BE COMPLETE
    pass


  def predict(self, in_path: str, out_path: str):
    """
    Function that should predict the angles for the sequence of nucleotides
    Args:
      - in_path: path to a `fasta` file.
        Example:
          "
          >17EP
          ACGUUCU
          "
      - out_path: path to a `.json` file where will be stored the prediciton.
        It should have the following keys: (example with `alpha` angle)
          {
            "17EP": {
              "sequence": "ACGUUCU",
              "angles": {"alpha": [0, 0, 0, 0, 0, 0, 0]}
            }

          }
    """
    x = convert_fasta_to_x(in_path)
    model = RandomModel(hidden_size=128)
    model.load_state_dict(torch.load("model.pt"))
    y_logits = model(x)
    y_pred = torch.argmax(y_logits, dim=-1)
    angles_preds = convert_y_to_angles(y_pred)
    save_prediction(angles_preds, out_path)
    return None

if __name__ == "__main__":
    # Example of usage
    in_path = os.path.join("data", "sample", "example.fasta")
    out_path = "sample.json"
    angle_helper = AngleHelper()
    angle_helper.predict(in_path, out_path)